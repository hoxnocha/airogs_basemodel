from typing import Any, Dict, List, Optional
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric, AUROC
from torchmetrics.functional import f1_score 
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url



class EfficientNetModule(LightningModule):
    def __init__(
            self, 
            
    ):
        super().__init__()
        self.save_hyperparameters()
        
      

        self.model = efficientnet_b4(pretrained=True)
        self.train_acc = MeanMetric()
        self.val_acc = MeanMetric()
        self.test_acc = MeanMetric()
        self.auroc = AUROC(num_classes=2)
        self.f1 = f1_score(num_classes=2)
                 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        self.train_acc(y_hat.softmax(dim=-1), y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        self.val_acc(y_hat.softmax(dim=-1), y)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=False)
        return loss
        
