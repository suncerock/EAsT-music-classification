import torch.utils.data as Data
import pytorch_lightning as pl

from .dataset import LatentSpaceDataset
from .featurizer.build_featurizer import build_featurizer


class LatentSpaceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_manifest_path,
        val_manifest_path,
        test_manifest_path,

        requires_vggish,
        requires_openl3,
        requires_passt,
        requires_pann,

        mixup,
        featurizer,

        batch_size,
        train_shuffle,
        num_workers
    ) -> None:
        super().__init__()

        self.train_dataset = LatentSpaceDataset(
            train_manifest_path,
            requires_vggish=requires_vggish,
            requires_openl3=requires_openl3,
            requires_passt=requires_passt,
            requires_pann=requires_pann,
            mixup=mixup,
            featurizer=build_featurizer(featurizer, training=True)
        )
        self.val_dataset = LatentSpaceDataset(
            val_manifest_path,
            requires_vggish=requires_vggish,
            requires_openl3=requires_openl3,
            requires_pann=requires_pann,
            requires_passt=requires_passt,
            mixup=0.0,
            featurizer=build_featurizer(featurizer, training=False)
        )
        self.test_dataset = LatentSpaceDataset(
            test_manifest_path,
            requires_vggish=requires_vggish,
            requires_openl3=requires_openl3,
            requires_pann=requires_pann,
            requires_passt=requires_passt,
            mixup=0.0,
            featurizer=build_featurizer(featurizer, training=False)
        )

        self.batch_size = batch_size
        self.train_shuffle = train_shuffle
        self.num_workers = num_workers

    def train_dataloader(self):
        return Data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.train_shuffle, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return Data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return Data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
