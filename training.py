"""
3D-CNN + LSTM multi-head model for hyperspectral patch-sequence classification.

This module defines a PyTorch model that:
  - Encodes each spectral–spatial patch (S bands × H × W) with a 3D-CNN encoder
    (spectral axis treated as depth/channel dimension for Conv3D).
  - Applies a lightweight channel-attention (Squeeze-and-Excitation) after the encoder
    to emphasize informative spectral–spatial channels.
  - Aggregates per-time-step encoded vectors into a temporal sequence and feeds them
    to a (Bi)LSTM to model progression across days.
  - Provides multiple output heads:
      * per-timestep 5-class classifier (healthy / pre_symp_disease / symp_disease /
        pre_symp_water / symp_water)
      * per-timestep anomaly score (sigmoid scalar) — useful for unsupervised cues
      * sequence-level early-alert (sigmoid) — predicts whether the leaf will become
        symptomatic within K days (learned from sequence)
  - Is written for clarity / research iteration: modular blocks, sensible defaults,
    and thorough docstring comments describing shapes and choices.

Design notes:
  - Input expected from DataLoader `patch_collate()` in prior module:
      batch = {
        "seq": Tensor shape (B, T, 1, S, H, W)
        "labels": (B, T) ...
      }
    We'll process the `seq` input: split or merge batch/time dims to run encoder.
  - Encoder produces a fixed-size vector per time-step (encoder_feat_dim).
  - LSTM is batch_first=True and returns per-timestep outputs which feed per-timestep heads.
  - The sequence-level alert head uses attention-pooled LSTM outputs (learned temporal attention).
"""

from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# Small utility blocks
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block (channel attention).
    Works on a 1D channel vector or on 3D conv channels. For our 3D conv output
    we will treat the conv output as (N, C, D, H, W) and apply SE across C.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(4, channels // reduction))
        self.fc2 = nn.Linear(max(4, channels // reduction), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, D, H, W) or (N, C)
        if x.dim() == 5:
            # global average pool spatial+spectral -> (N, C)
            n, c, d, h, w = x.size()
            y = x.mean(dim=[2, 3, 4])  # average over D,H,W
        else:
            y = x
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        if x.dim() == 5:
            # expand to multiply
            y = y.view(n, c, 1, 1, 1)
            return x * y
        else:
            return x * y


class Conv3DBlock(nn.Module):
    """
    A small Conv3D -> BN -> ReLU block with optional spatial pooling.
    Kernel shapes use (kernel_depth, kernel_h, kernel_w) so we can specify a
    smaller spectral kernel vs spatial kernel if desired.
    """
    def __init__(self, in_ch: int, out_ch: int,
                 kernel: Tuple[int, int, int] = (3, 3, 3),
                 pool: Optional[Tuple[int, int, int]] = None):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel, padding=tuple(k // 2 for k in kernel))
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(pool) if pool is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


# Main model
class Encoder3D(nn.Module):
    """
    3D-CNN encoder for spectral-spatial patches.

    Input (per time-step):  tensor shape (N, C=1, S, H, W)
      - we treat spectral axis as spatial depth S for Conv3d.
    Output: feature vector per sample: (N, encoder_dim)

    Architecture (example / recommended):
      - ConvBlock(1 -> 16, kernel=(3,3,3), pool=(1,2,2))  # keep spectral dim same, reduce spatial
      - ConvBlock(16->32, kernel=(3,3,3), pool=(1,2,2))
      - ConvBlock(32->64, kernel=(3,3,3), pool=(1,2,2))
      - Global avg pool over (D,H,W) -> flatten -> linear -> encoder_dim
      - SE block applied before pooling to reweight channels
    """
    def __init__(self, in_channels: int = 1, encoder_dim: int = 256, base_channels: int = 16):
        super().__init__()
        # three conv blocks with spatial pooling only (preserve spectral depth early)
        self.block1 = Conv3DBlock(in_channels, base_channels, kernel=(3, 3, 3), pool=(1, 2, 2))
        self.block2 = Conv3DBlock(base_channels, base_channels * 2, kernel=(3, 3, 3), pool=(1, 2, 2))
        self.block3 = Conv3DBlock(base_channels * 2, base_channels * 4, kernel=(3, 3, 3), pool=(1, 2, 2))
        self.se = SEBlock(base_channels * 4, reduction=8)
        # after block3 the spatial dims are reduced by 2*2*2 = 8 in H/W if starting patch 32 -> 4
        # We'll global average pool across spectral and spatial dims to get a channel vector
        self.fc = nn.Linear(base_channels * 4, encoder_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 1, S, H, W)
        returns: (N, encoder_dim)
        """
        x = self.block1(x)   # -> (N, C1, S, H/2, W/2)
        x = self.block2(x)   # -> (N, C2, S, H/4, W/4)
        x = self.block3(x)   # -> (N, C3, S, H/8, W/8)
        x = self.se(x)       # channel attention
        # global average pool over spectral + spatial dims -> (N, C3)
        x = x.mean(dim=[2, 3, 4])
        x = self.fc(x)
        x = F.relu(x)
        return x


class TemporalAttention(nn.Module):
    """
    Simple attention module over temporal LSTM outputs.
    Input: H (N, T, hidden)
    Output: context vector (N, hidden)
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # h: (N, T, H)
        scores = self.attn(h).squeeze(-1)  # (N, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # (N, T)
        weights = weights.unsqueeze(-1)  # (N, T, 1)
        context = (h * weights).sum(dim=1)  # (N, H)
        return context, weights.squeeze(-1)


class Hybrid3DConvLSTM(nn.Module):
    """
    Full model that combines:
      - 3D-CNN encoder -> per-time-step features
      - LSTM temporal module over sequence of features
      - Multi-head outputs:
          * per-timestep classifier (5-way)
          * per-timestep anomaly (sigmoid)
          * sequence-level early alert (sigmoid from attention-pooled LSTM)
    """

    def __init__(self,
                 encoder_params: Dict = None,
                 lstm_hidden: int = 256,
                 lstm_layers: int = 1,
                 bidirectional: bool = True,
                 classifier_hidden: int = 128,
                 num_classes: int = 5,
                 dropout: float = 0.3,
                 use_attention: bool = True):
        super().__init__()
        enc_p = encoder_params or {}
        self.encoder = Encoder3D(**enc_p)  # default enc_p: in_channels=1, encoder_dim=256
        enc_dim = enc_p.get("encoder_dim", 256)
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM temporal head: receives per-timestep encoder vectors
        self.lstm = nn.LSTM(input_size=enc_dim,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout if lstm_layers > 1 else 0.0)

        lstm_out_dim = lstm_hidden * self.num_directions

        # Per-timestep classifier head -> applied to each timestep output of LSTM
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes)
        )

        # Per-timestep anomaly score head (single scalar)
        self.anomaly_head = nn.Sequential(
            nn.Linear(lstm_out_dim, classifier_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden // 2, 1)
        )

        # Sequence-level alert: attention pool over time then small MLP -> sigmoid
        self.use_attention = use_attention
        if use_attention:
            self.temporal_attention = TemporalAttention(lstm_out_dim)
            self.alert_mlp = nn.Sequential(
                nn.Linear(lstm_out_dim, lstm_out_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(lstm_out_dim // 2, 1)
            )
        else:
            # global pooling alternative
            self.alert_mlp = nn.Sequential(
                nn.Linear(lstm_out_dim, lstm_out_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(lstm_out_dim // 2, 1)
            )

        # initialization helpers (optional)
        self._init_weights()

    def _init_weights(self):
        # small init for linear layers (Xavier)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, seq: torch.Tensor, label_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
          seq: input tensor shape (B, T, 1, S, H, W)
          label_mask: optional bool tensor shape (B, T) indicating valid labelled timesteps;
                      used by attention to ignore padded/missing steps.

        Returns a dict with:
          'logits': (B, T, num_classes)  -- raw logits for per-timestep classification
          'probs': (B, T, num_classes)   -- softmax probabilities
          'anomaly_logits': (B, T, 1)    -- raw scalar for anomaly
          'anomaly_probs': (B, T, 1)     -- sigmoid anomaly scores
          'alert_logit': (B, 1)          -- sequence-level alert logit
          'alert_prob': (B, 1)           -- sequence-level alert probability
          'lstm_outputs': (B, T, lstm_out_dim) internal for debugging/auxiliary use
        """
        B, T, C, S, H, W = seq.shape
        # Merge batch and time dims to run encoder efficiently
        x = seq.view(B * T, C, S, H, W)  # (B*T, 1, S, H, W)
        # pass through encoder: returns (B*T, enc_dim)
        enc = self.encoder(x)  # (B*T, enc_dim)
        # reshape back to sequence form
        enc_seq = enc.view(B, T, -1)  # (B, T, enc_dim)

        # LSTM expects (B, T, enc_dim) when batch_first=True
        lstm_out, _ = self.lstm(enc_seq)  # lstm_out: (B, T, lstm_out_dim)
        # Per-timestep classification & anomaly
        logits = self.classifier(lstm_out)  # (B, T, num_classes)
        probs = torch.softmax(logits, dim=-1)
        anomaly_logits = self.anomaly_head(lstm_out)  # (B, T, 1)
        anomaly_probs = torch.sigmoid(anomaly_logits)

        # Sequence-level alert
        if self.use_attention:
            # create boolean mask for attention: True where label_mask True; if None use all True
            if label_mask is not None:
                attn_mask = label_mask  # (B, T) boolean
            else:
                attn_mask = torch.ones((B, T), dtype=torch.bool, device=seq.device)
            context, attn_weights = self.temporal_attention(lstm_out, mask=attn_mask)
            alert_logit = self.alert_mlp(context).squeeze(-1)  # (B,)
        else:
            # simple mean pooling over time (only valid timesteps)
            if label_mask is not None:
                # replace masked positions with 0 and average by count
                maskf = label_mask.float().unsqueeze(-1)
                summed = (lstm_out * maskf).sum(dim=1)
                denom = maskf.sum(dim=1).clamp(min=1.0)
                pooled = summed / denom
            else:
                pooled = lstm_out.mean(dim=1)
            alert_logit = self.alert_mlp(pooled).squeeze(-1)  # (B,)

        alert_prob = torch.sigmoid(alert_logit).unsqueeze(-1)  # (B, 1)
        alert_logit = alert_logit.unsqueeze(-1)

        return {
            "logits": logits,                     # (B, T, num_classes)
            "probs": probs,                       # (B, T, num_classes)
            "anomaly_logits": anomaly_logits,     # (B, T, 1)
            "anomaly_probs": anomaly_probs,       # (B, T, 1)
            "alert_logit": alert_logit,           # (B, 1)
            "alert_prob": alert_prob,             # (B, 1)
            "lstm_outputs": lstm_out              # (B, T, lstm_out_dim)
        }


# Example quick test (not executed here)
if __name__ == "__main__":
    # quick sanity check on tensor shapes (toy run)
    B, T, C, S, H, W = 2, 5, 1, 32, 32, 32
    toy_in = torch.randn(B, T, C, S, H, W)
    model = Hybrid3DConvLSTM(encoder_params={"in_channels": 1, "encoder_dim": 256, "base_channels": 16},
                             lstm_hidden=128, lstm_layers=1, bidirectional=True, classifier_hidden=128)
    out = model(toy_in)  # label_mask omitted for simple test
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
