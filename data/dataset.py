import json
import soundfile as sf

import numpy as np
import torch
import torch.utils.data as Data

class LatentSpaceDataset(Data.Dataset):
    def __init__(
        self,
        manifest_path,
        sr=16000,

        requires_vggish=False,
        requires_openl3=False,
        requires_pann=False,
        requires_passt=False,

        mixup=0.0,

        featurizer=None
    ) -> None:
        super().__init__()

        with open(manifest_path) as f:
            self.data = [json.loads(line) for line in f]

        self.featurizer = featurizer
        self.sr = sr

        self.requires_vggish = requires_vggish
        self.requires_openl3 = requires_openl3
        self.requires_pann = requires_pann
        self.requires_passt = requires_passt

        self.mixup = mixup

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.mixup > 0:
            return self._transform_data(self.data[idx])

        mixup_idx = np.random.randint(low=0, high=len(self.data))
        mixup_lam = np.random.beta(self.mixup, self.mixup)
        mixup_lam = max(mixup_lam, 1 - mixup_lam)

        return self._mixup_data(
            self._transform_data(self.data[idx]),
            self._transform_data(self.data[mixup_idx]),
            mixup_lam
            )

    def _transform_data(self, data):
        output_data = dict()

        audio, _ = sf.read(data["audio_path"])
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        output_data["x"] = self.featurizer(torch.from_numpy(audio).float())

        y = torch.tensor(data["label"], dtype=torch.float32)

        output_data["y"] = torch.where(y == 1, torch.ones_like(y), torch.zeros_like(y))
        output_data["y_mask"] = (y != 0).bool()

        if self.requires_vggish:
            feature = torch.from_numpy(np.load(data["vggish"]))
            output_data["vggish"] = (feature.float() - 128) / 256
        if self.requires_openl3:
            feature = torch.from_numpy(np.load(data["openl3"]))
            output_data["openl3"] = feature.float() - 2.24
        if self.requires_pann:
            output_data["pann"] = torch.from_numpy(np.load(data["pann"]))
        if self.requires_passt:
            output_data["passt"] = torch.from_numpy(np.load(data["passt"]))

        return output_data

    def _mixup_data(self, data1, data2, lam):
        output_data = dict()
        output_data["x"] = data1["x"] * lam + data2["x"] * (1. - lam)
        output_data["y"] = data1["y"] * lam + data2["y"] * (1. - lam)
        output_data["y_mask"] = data1["y_mask"] | data2["y_mask"]
        if self.requires_vggish:
            output_data["vggish"] = data1["vggish"] * lam + data2["vggish"] * (1. - lam)
        if self.requires_openl3:
            output_data["openl3"] = data1["openl3"] * lam + data2["openl3"] * (1. - lam)
        if self.requires_pann:
            output_data["pann"] = data1["pann"] * lam + data2["pann"] * (1. - lam)
        if self.requires_passt:
            output_data["passt"] = data1["passt"] * lam + data2["passt"] * (1. - lam)
        
        return output_data