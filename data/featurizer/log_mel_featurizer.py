import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

EPSILON = 1e-5

class LogMelFeaturizer(nn.Module):
    def __init__(self, n_fft=2048, hop_length=512, win_length=1024, sr=44100,
        n_mels=128, fmin=20.0, fmax=22050.0, fmin_aug_range=0, fmax_aug_range=0,
        normalize_mean=0.0, normalize_std=1.0,
        freqm=48, timem=192, training=True
    ) -> None:
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sr = sr

        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range

        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        self.training = training
        self.freqm = torchaudio.transforms.FrequencyMasking(freqm) if (freqm > 0 and training) else nn.Identity()
        self.timem = torchaudio.transforms.TimeMasking(timem) if (timem > 0 and training) else nn.Identity()

        window = torch.hann_window(win_length, periodic=False)
        self.register_buffer("window", window, persistent=False)

        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-0.97, 1]]]), persistent=False)

    def forward(self, x):
        with torch.no_grad():
            x = x.unsqueeze(dim=0)
            x = F.conv1d(x.unsqueeze(dim=1), self.preemphasis_coefficient).squeeze(dim=1)

            x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window, return_complex=True)
            x = torch.abs(x) ** 2

            fmin = self.fmin + np.random.randint(0, self.fmin_aug_range) if self.training and self.fmin_aug_range > 0 else self.fmin
            fmax = self.fmax - np.random.randint(0, self.fmax_aug_range) if self.training and self.fmin_aug_range > 0 else self.fmax
            mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
                self.n_mels, self.n_fft, self.sr, fmin, fmax, vtln_low=100, vtln_high=-500, vtln_warp_factor=1.0)
            mel_basis = F.pad(mel_basis, (0, 1), mode='constant', value=0)
        
            mel_spec = torch.matmul(mel_basis, x)
            mel_spec = torch.log(mel_spec + EPSILON)

            if self.training:
                mel_spec = self.freqm(mel_spec)
                mel_spec = self.timem(mel_spec)
            mel_spec = mel_spec.squeeze(dim=0)

        return (mel_spec - self.normalize_mean) / self.normalize_std
