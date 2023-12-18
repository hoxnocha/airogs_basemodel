from pandas import DataFrame
import lightning.pytorch as pl
from airogs_basemodel.data.dataset import AirogsDataset
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data.dataloader import DataLoader, default_collate


class AirogsDataModule(pl.LightningDataModule):
    def __init__(self, train_batch_size: int, test_batch_size: int, num_workers: int):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.

    def setup(self, stage: [str] = None) -> None:
        """Setup the data module.

        Args:
            stage (Optional[str], optional): Stage of the data module setup. Defaults to None.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = AirogsDataset(task="classification", transform=self.train_transform)
            self.val_dataset = AirogsDataset(task="classification", transform=self.val_transform)
        if stage == "test" or stage is None:
            self.test_dataset = AirogsDataset(task="classification", transform=self.test_transform)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Get the train dataloader.

        Returns:
            TRAIN_DATALOADERS: Train dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            
           
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Get the validation dataloader.

        Returns:
            EVAL_DATALOADERS: Validation dataloader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Get the test dataloader.

        Returns:
            EVAL_DATALOADERS: Test dataloader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            
        )