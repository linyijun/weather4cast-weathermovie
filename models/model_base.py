import numpy as np

from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torch


class ModelBase(pl.LightningModule):
    
    def __init__(
        self,
        lr=0.01,
        weight_decay=0.0001,
        metric='mse',
    ):
        
        super(ModelBase, self).__init__()
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.metric = metric
    
        self.loss_fn = {'smoothL1': nn.SmoothL1Loss(), 
                        'L1': nn.L1Loss(), 
                        'mse': F.mse_loss}
        self.loss_fn = self.loss_fn[metric]
            
    def forward(self, x):
        pass
    
    def _compute_loss(self, y_hat, y, agg=True):
        if agg:
            loss = self.loss_fn(y_hat, y)
        else:
            loss = self.loss_fn(y_hat, y, reduction='none')
        return loss
    
    def training_step(self, batch, batch_idx):
        pass
        
    def validation_step(self, batch, batch_idx):
        pass
          
    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        
        return {"optimizer": optimizer}
