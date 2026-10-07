"""Causal CNN + GRU model for streaming double-tap detection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    1D convolution with strict causality: output at time t
    depends only on inputs at times <= t.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )
        self.left_pad = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class CausalConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class CausalCNNGRU(nn.Module):
    """
    Causal CNN front-end + GRU for IMU double-tap recognition.

    Input:  [B, T, F]  F=6 IMU channels
    Output: [B, T, C]  per-frame logits, C=3 (none, left, right)
    """

    def __init__(
        self,
        input_dim: int = 6,
        num_classes: int = 3,
        cnn_channels: int = 32,
        gru_hidden: int = 64,
        gru_layers: int = 1,
        kernel_size: int = 5,
        dilations: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers

        self.input_proj = nn.Linear(input_dim, cnn_channels)

        blocks = []
        for d in dilations:
            blocks.append(CausalConvBlock(cnn_channels, kernel_size, d))
        self.cnn = nn.Sequential(*blocks)

        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            cnn_channels,
            gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, num_classes),
        )

        self._receptive_field = 1 + sum(
            (kernel_size - 1) * d for d in dilations
        )

    @property
    def receptive_field(self) -> int:
        return self._receptive_field

    def _gru_one_step(
        self, x: torch.Tensor, h: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the GRU gates for one frame with the weights from self.gru.

        Explicit gates avoid an ONNX GRU operator whose MindSpore Lite
        conversion does not reproduce the recurrent state of PyTorch's GRU.
        """
        if h is None:
            h = x.new_zeros(self.gru_layers, x.shape[0], self.gru_hidden)

        next_states = []
        for layer in range(self.gru_layers):
            prev = h[layer]
            input_gates = F.linear(
                x,
                getattr(self.gru, f"weight_ih_l{layer}"),
                getattr(self.gru, f"bias_ih_l{layer}"),
            )
            hidden_gates = F.linear(
                prev,
                getattr(self.gru, f"weight_hh_l{layer}"),
                getattr(self.gru, f"bias_hh_l{layer}"),
            )
            ir, iz, inn = input_gates.chunk(3, dim=-1)
            hr, hz, hn = hidden_gates.chunk(3, dim=-1)
            reset = torch.sigmoid(ir + hr)
            update = torch.sigmoid(iz + hz)
            candidate = torch.tanh(inn + reset * hn)
            x = (1.0 - update) * candidate + update * prev
            next_states.append(x)

        return x.unsqueeze(1), torch.stack(next_states)

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:  [B, T, F]
            h0: [L, B, H] optional initial GRU state

        Returns:
            logits: [B, T, C]
            h_n:    [L, B, H] final GRU state
        """
        b, t, _ = x.shape
        cnn_in = self.input_proj(x)  # [B, T, C]
        cnn_in = cnn_in.transpose(1, 2)  # [B, C, T]
        cnn_out = self.cnn(cnn_in).transpose(1, 2)  # [B, T, C]
        cnn_out = self.dropout(cnn_out)

        gru_out, h_n = self.gru(cnn_out, h0)
        logits = self.head(gru_out)
        return logits, h_n

    def step(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor | None,
        cnn_buffer: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process a single time step for online inference (model.eval()).

        Args:
            x_t: [B, 1, F] or [B, F] — one IMU sample
            h:   [L, B, H] GRU state
            cnn_buffer: [B, C, R] packed input histories for each conv block;
                block i stores (kernel_size - 1) * dilation_i frames.
                The final slot is reserved to keep the exported state shape.

        Returns:
            prob: [B, 1, C] softmax P(none, left, right)
            h_new: [L, B, H]
            buffer_new: [B, C_cnn, R]
        """
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)
        elif x_t.dim() != 3 or x_t.shape[1] != 1:
            raise ValueError(
                f"step() expects x_t shape [B, 1, F] or [B, F], got {tuple(x_t.shape)}"
            )

        proj = self.input_proj(x_t).transpose(1, 2)  # [B, C, 1]

        if cnn_buffer is None:
            cnn_buffer = proj.new_zeros(
                proj.shape[0], proj.shape[1], self._receptive_field
            )

        # A conv block needs its *own input* history: replaying projected input
        # through the whole CNN would invent outputs before the first sample.
        # Keeping the histories separately reproduces forward()'s left padding.
        current = proj
        offset = 0
        histories = []
        for block in self.cnn:
            n_past = block.conv.left_pad
            if n_past:
                past = cnn_buffer[:, :, offset : offset + n_past]
                histories.append(torch.cat([past[:, :, 1:], current], dim=2))
                current = block(torch.cat([past, current], dim=2))[:, :, -1:]
            else:
                current = block(current)
            offset += n_past

        # One reserved slot preserves the [B, C, receptive_field] I/O of the
        # existing ONNX/MindSpore step model and its HarmonyOS caller.
        buffer_new = torch.cat([*histories, cnn_buffer[:, :, -1:]], dim=2)
        cnn_out = self.dropout(current.transpose(1, 2))  # [B, 1, C]
        gru_out, h_new = self._gru_one_step(cnn_out[:, 0, :], h)
        logit = self.head(gru_out)  # [B, 1, num_classes]
        prob = torch.softmax(logit, dim=-1)
        return prob, h_new, buffer_new
