"""
model.py — CRNN Architecture for Handwritten Character Recognition
==================================================================
Architecture:
  Input (1 × 32 × W)
    └─ CNN Backbone   7 convolutional blocks → feature map (512 × 1 × W')
    └─ Map-to-Seq     squeeze height → sequence of 512-dim vectors
    └─ BiLSTM ×2      bidirectional LSTM, hidden=256  → sequence of class logits
    └─ CTC Loss       alignment-free sequence learning

num_classes = 63  (62 EMNIST ByClass characters + 1 CTC blank)
"""

import torch
import torch.nn as nn
from dataset import NUM_CLASSES          # 63


class BidirectionalLSTM(nn.Module):
    def __init__(self, in_size, hidden, out_size):
        super().__init__()
        self.lstm   = nn.LSTM(in_size, hidden, bidirectional=True, batch_first=False)
        self.linear = nn.Linear(hidden * 2, out_size)

    def forward(self, x):                # x: (T, B, in_size)
        out, _ = self.lstm(x)            # (T, B, hidden*2)
        T, B, H = out.shape
        out = self.linear(out.view(T * B, H))
        return out.view(T, B, -1)


class CRNN(nn.Module):
    """
    CNN–RNN Hybrid network.

    Parameters
    ----------
    num_classes : vocabulary size + 1 blank  (default 63 for EMNIST ByClass)
    hidden_size : LSTM hidden units per direction  (default 256)
    """

    def __init__(self, num_classes=NUM_CLASSES, hidden_size=256):
        super().__init__()

        # ── CNN Backbone ──────────────────────────────────────────────
        def conv_bn_relu(in_c, out_c, k=3, s=1, p=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, k, s, p, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.cnn = nn.Sequential(
            # Block 1  64 ch   H/2
            conv_bn_relu(1,   64),
            nn.MaxPool2d(2, 2),

            # Block 2  128 ch  H/4
            conv_bn_relu(64,  128),
            nn.MaxPool2d(2, 2),

            # Block 3+4  256 ch  H/8
            conv_bn_relu(128, 256),
            conv_bn_relu(256, 256),
            nn.MaxPool2d((2, 1), (2, 1)),

            # Block 5+6  512 ch  H/16
            conv_bn_relu(256, 512),
            conv_bn_relu(512, 512),
            nn.MaxPool2d((2, 1), (2, 1)),

            # Block 7  collapse remaining height (32 → 1)
            nn.Conv2d(512, 512, kernel_size=2, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # ── Sequence Modeling ─────────────────────────────────────────
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, num_classes),
        )

    def forward(self, x):
        # x : (B, 1, H, W)
        feat = self.cnn(x)                  # (B, 512, 1, W')
        assert feat.size(2) == 1, (
            f"CNN height must be 1 after conv stack, got {feat.size(2)}. "
            "Ensure input height = 32."
        )
        feat = feat.squeeze(2)              # (B, 512, W')
        feat = feat.permute(2, 0, 1)        # (W', B, 512)  — time-first for LSTM
        logits = self.rnn(feat)             # (W', B, num_classes)
        return logits


def build_model(device="cpu"):
    model = CRNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] CRNN  params={n_params:,}  classes={NUM_CLASSES}  device={device}")
    return model


# ── Quick architecture test ───────────────────────────────────────────────────
if __name__ == "__main__":
    model = build_model()
    dummy = torch.zeros(4, 1, 32, 32)        # batch=4, single-char (EMNIST)
    out   = model(dummy)
    print(f"Input  : {tuple(dummy.shape)}")
    print(f"Output : {tuple(out.shape)}   (T, B, classes)")
    print("model.py check ✓")
