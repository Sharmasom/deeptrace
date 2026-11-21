"""Model family. Input x: uint8 [B,T,3,224,224]. Output: one logit per clip [B].

backbone : EfficientNet-B0 (ImageNet), blocks < unfreeze_from frozen with BatchNorm
           kept in eval mode; per frame -> 1280-d after global average pooling.
freq     : optional (G4). Per frame: grey image -> 2-D FFT -> log magnitude,
           centred, -> small CNN -> 128-d. Concatenated to the RGB features.
heads    : mean        v1  mean over time (no temporal model)
           bilstm      v2a BiLSTM(256) + attention pooling over time
           tdiff       v2b lightweight temporal-difference head: features and their
                       frame-to-frame differences -> two 1-D temporal convs -> pool.
                       Inspired by FTCN's focus on temporal incoherence; NOT FTCN.

Tensor flow (B clips, T frames):
  [B,T,3,224,224] -> [B*T,3,224,224] -> EffNet -> [B*T,1280,7,7] -> GAP -> [B*T,1280]
  (+ freq [B*T,128]) -> [B,T,D] -> head -> [B,H] -> dropout -> linear -> [B]
"""
import torch
import torch.nn as nn
from torchvision import models

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class FreqBranch(nn.Module):
    def __init__(self, out=128):
        super().__init__()
        c = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, 2, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.net = nn.Sequential(c(1, 16), c(16, 32), c(32, 64), c(64, 128), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(128, out)

    def forward(self, x01):                       # x01: [N,3,H,W] in [0,1]
        g = (0.299 * x01[:, 0] + 0.587 * x01[:, 1] + 0.114 * x01[:, 2]).float()
        spec = torch.fft.fftshift(torch.fft.fft2(g), dim=(-2, -1)).abs()
        spec = torch.log1p(spec)
        spec = (spec - spec.mean((-2, -1), keepdim=True)) / (spec.std((-2, -1), keepdim=True) + 1e-6)
        return self.fc(self.net(spec.unsqueeze(1)).flatten(1))


class TDiffHead(nn.Module):
    def __init__(self, d, h=256):
        super().__init__()
        self.proj = nn.Linear(2 * d, h)
        self.conv = nn.Sequential(nn.Conv1d(h, h, 3, padding=1), nn.GELU(), nn.Conv1d(h, h, 3, padding=1), nn.GELU())
        self.out_dim = h

    def forward(self, f):                         # [B,T,D]
        diff = torch.cat([torch.zeros_like(f[:, :1]), f[:, 1:] - f[:, :-1]], 1)
        z = self.proj(torch.cat([f, diff], -1)).transpose(1, 2)     # [B,h,T]
        z = self.conv(z)
        return z.mean(-1)                          # [B,h]


class BiLSTMHead(nn.Module):
    def __init__(self, d, h=256):
        super().__init__()
        self.rnn = nn.LSTM(d, h, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * h, 1)
        self.out_dim = 2 * h

    def forward(self, f):
        o, _ = self.rnn(f)
        return (torch.softmax(self.att(o), 1) * o).sum(1)


class MeanHead(nn.Module):
    def __init__(self, d):
        super().__init__(); self.out_dim = d

    def forward(self, f):
        return f.mean(1)


class Net(nn.Module):
    def __init__(self, head="mean", freq=False, unfreeze_from=5, dropout=0.4, pretrained=True):
        super().__init__()
        eff = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        self.backbone, self.unfreeze_from = eff.features, unfreeze_from
        for i, blk in enumerate(self.backbone):
            for p in blk.parameters():
                p.requires_grad = i >= unfreeze_from
        self.freq = FreqBranch() if freq else None
        d = 1280 + (128 if freq else 0)
        self.head = {"mean": lambda: MeanHead(d), "bilstm": lambda: BiLSTMHead(d),
                     "tdiff": lambda: TDiffHead(d)}[head]()
        self.drop, self.fc = nn.Dropout(dropout), nn.Linear(self.head.out_dim, 1)
        self.register_buffer("mean", MEAN.clone()); self.register_buffer("std", STD.clone())

    def train(self, mode=True):
        super().train(mode)
        for i, blk in enumerate(self.backbone):
            if i < self.unfreeze_from:
                blk.eval()
        return self

    def frame_features(self, x):
        B, T = x.shape[:2]
        x01 = x.flatten(0, 1).float() / 255
        f = self.backbone((x01 - self.mean) / self.std).mean((2, 3))
        if self.freq is not None:
            with torch.autocast("cuda", enabled=False):
                f = torch.cat([f.float(), self.freq(x01)], 1)
        return f.view(B, T, -1)

    def forward(self, x):
        return self.fc(self.drop(self.head(self.frame_features(x)))).squeeze(1)
