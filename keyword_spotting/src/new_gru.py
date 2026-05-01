from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NewGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for p in self.parameters():
            nn.init.uniform_(p, -stdv, stdv)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gi = F.linear(x, self.weight_ih, self.bias_ih)
        gh = F.linear(h, self.weight_hh, self.bias_hh)

        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)

        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)

        return (1.0 - z) * n + z * h


class NewGRU(nn.Module):
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
            raise ValueError("NewGRU only supports batch_first=True")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout

        cells = []
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            cell = NewGRUCell(in_size, hidden_size)
            cells.append(cell)
            self.register_parameter(f"weight_ih_l{layer}", cell.weight_ih)
            self.register_parameter(f"weight_hh_l{layer}", cell.weight_hh)
            self.register_parameter(f"bias_ih_l{layer}", cell.bias_ih)
            self.register_parameter(f"bias_hh_l{layer}", cell.bias_hh)
        # Plain list — parameters are already owned by NewGRU via register_parameter above,
        # so using nn.ModuleList would double-register them under different key names.
        self.cells = cells

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        batch, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch, self.hidden_size, device=device, dtype=dtype)

        layer_input = x
        h_n_layers = []

        for layer, cell in enumerate(self.cells):
            h = h0[layer]
            outputs = []
            for t in range(seq_len):
                h = cell(layer_input[:, t, :], h)
                outputs.append(h)
            layer_output = torch.stack(outputs, dim=1)
            h_n_layers.append(h)

            if layer < self.num_layers - 1 and self.dropout > 0.0 and self.training:
                layer_output = F.dropout(layer_output, p=self.dropout, training=True)

            layer_input = layer_output

        h_n = torch.stack(h_n_layers, dim=0)
        return layer_input, h_n


def from_torch_gru(gru: nn.GRU) -> NewGRU:
    if not gru.batch_first:
        raise ValueError("Source nn.GRU must have batch_first=True")
    if gru.bidirectional:
        raise ValueError("Bidirectional GRU is not supported")

    new = NewGRU(
        input_size=gru.input_size,
        hidden_size=gru.hidden_size,
        num_layers=gru.num_layers,
        batch_first=True,
        dropout=gru.dropout,
    )

    with torch.no_grad():
        for layer in range(gru.num_layers):
            for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                src = getattr(gru, f"{name}_l{layer}")
                dst = getattr(new, f"{name}_l{layer}")
                dst.copy_(src)

    return new
