from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from new_gru import NewGRU
from .quant_utils import FixedScaleQuantizer, Quantizer


class QuantizedNewGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))

        self.x_quant = Quantizer(observer_type="percentile")
        self.w_ih_quant = Quantizer(observer_type="max")
        self.w_hh_quant = Quantizer(observer_type="max")
        self.gi_quant = Quantizer(observer_type="percentile")
        self.gh_quant = Quantizer(observer_type="percentile")
        self.h_quant = Quantizer(observer_type="percentile")
        self.sig_r_quant = FixedScaleQuantizer(range_max=1.0)
        self.sig_z_quant = FixedScaleQuantizer(range_max=1.0)
        self.tanh_n_quant = FixedScaleQuantizer(range_max=1.0)


class QuantizedNewGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not batch_first:
            raise ValueError("QuantizedNewGRU only supports batch_first=True")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout

        cells = []
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            cell = QuantizedNewGRUCell(in_size, hidden_size)
            cells.append(cell)
        self.cells = nn.ModuleList(cells)

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        return_sequences: bool = True,
    ):
        batch, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch, self.hidden_size, device=device, dtype=dtype)

        layer_input = x
        h_n_layers = []

        for layer, cell in enumerate(self.cells):
            h = h0[layer]
            is_last_layer = layer == self.num_layers - 1
            collect_outputs = return_sequences or not is_last_layer

            x_q = cell.x_quant(layer_input)
            w_ih_q = cell.w_ih_quant(cell.weight_ih)
            gi_all = F.linear(x_q, w_ih_q, cell.bias_ih)
            gi_all = cell.gi_quant(gi_all)

            outputs: List[torch.Tensor] = []
            for t in range(seq_len):
                w_hh_q = cell.w_hh_quant(cell.weight_hh)
                gh = F.linear(h, w_hh_q, cell.bias_hh)
                gh = cell.gh_quant(gh)

                gi = gi_all[:, t]
                i_r, i_z, i_n = gi.chunk(3, dim=-1)
                h_r, h_z, h_n = gh.chunk(3, dim=-1)

                r = cell.sig_r_quant(torch.sigmoid(i_r + h_r))
                z = cell.sig_z_quant(torch.sigmoid(i_z + h_z))
                n = cell.tanh_n_quant(torch.tanh(i_n + r * h_n))

                h_new = (1.0 - z) * n + z * h
                h = cell.h_quant(h_new)

                if collect_outputs:
                    outputs.append(h)

            if not collect_outputs:
                layer_output = h.unsqueeze(1)
            else:
                layer_output = torch.stack(outputs, dim=1)
            h_n_layers.append(h)

            if layer < self.num_layers - 1 and self.dropout > 0.0 and self.training:
                layer_output = F.dropout(layer_output, p=self.dropout, training=True)

            layer_input = layer_output

        h_n = torch.stack(h_n_layers, dim=0)
        if return_sequences:
            return layer_input, h_n
        return h_n[-1], h_n


def from_new_gru(src: NewGRU) -> QuantizedNewGRU:
    new = QuantizedNewGRU(
        input_size=src.input_size,
        hidden_size=src.hidden_size,
        num_layers=src.num_layers,
        batch_first=True,
        dropout=src.dropout,
    )
    with torch.no_grad():
        for layer, (src_cell, dst_cell) in enumerate(zip(src.cells, new.cells)):
            dst_cell.weight_ih.copy_(src_cell.weight_ih)
            dst_cell.weight_hh.copy_(src_cell.weight_hh)
            dst_cell.bias_ih.copy_(src_cell.bias_ih)
            dst_cell.bias_hh.copy_(src_cell.bias_hh)
    return new
