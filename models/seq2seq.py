import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from models.ConvLSTMCell import ConvLSTMCell
from models.model_base import ModelBase


class EncoderDecoderConvLSTM(ModelBase):
    
    def __init__(
        self, 
        in_channels: int, 
        h_channels: int, 
        out_channels: int, 
        n_encoder_layers: int=1, 
        n_decoder_layers: int=1,
        kernel_size: tuple=(3, 3),
        dropout: float=0.1,
        lr: float=1e-4,
        weight_decay: float=1e-4
    ):
        
        super().__init__(lr=lr,
                         weight_decay=weight_decay)
        
        self.save_hyperparameters()
        self.kernel_size = kernel_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        
        self.encoder = ConvLSTMCell(in_channels=in_channels,
                                    h_channels=h_channels,
                                    kernel_size=kernel_size)

        self.decoder = ConvLSTMCell(in_channels=out_channels,
                                    h_channels=h_channels,
                                    kernel_size=kernel_size)

#         self.predict = nn.Sequential(
#             nn.Conv2d(in_channels=h_channels,
#                       out_channels=h_channels,
#                       kernel_size=(1, 1),
#                       padding=(0, 0)),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=h_channels,
#                       out_channels=out_channels,
#                       kernel_size=(1, 1),
#                       padding=(0, 0)),
#         )
        
        self.predict = nn.Conv2d(in_channels=h_channels,
                                 out_channels=out_channels,
                                 kernel_size=(1, 1),
                                 padding=(0, 0))
       
        self.do = nn.Dropout(p=self.dropout)

    def encode(self, x, seq_len, hidden_state, cell_state):
        
        for t in range(seq_len):
            
            hidden_state, cell_state = self.encoder(input_data=x[:, t, :, :], 
                                                    prev_state=[hidden_state, cell_state])
        return hidden_state, cell_state  # [b, h_dim, h, w]
                
                
    def decode(self, lag_y, horizon, hidden_state, cell_state, teacher_forcing_ratio=0.5):
        
        out = lag_y[:, 0, :, :, :]  # out: [b, out_dim, h, w]
        
        outputs = []
        for t in range(horizon):
            
            if random.random() < teacher_forcing_ratio:
                hidden_state, cell_state = self.decoder(input_data=lag_y[:, t, :, :, :], 
                                                        prev_state=[hidden_state, cell_state])
            else:
                hidden_state, cell_state = self.decoder(input_data=out, 
                                                        prev_state=[hidden_state, cell_state])  
            out = self.predict(hidden_state)
            outputs += [out]
            
        outputs = torch.stack(outputs, dim=1) 
        return outputs  # [b, horizon, out_dim, h, w]
    
    def forward(self, x, y, teacher_forcing_ratio=0.5):

        """
        params:
            x (5-D Tensor) - [b, t, c, h, w]        #   batch, seq_len, channel, height, width
            y (5-D Tensor) - [b, t, c, h, w]        #   batch, horizon, channel, height, width
        """

        # find size of different input dimensions
        b, seq_len, x_c, h, w = x.size()
        _, horizon, y_c, _, _ = y.size()

        # construct lag_y by taking the last step of x and first steps from y
        in_x = min(x_c, y_c)
        lag_y = torch.cat([x[:, -1:, :in_x, :, :], y[:, :-1, :in_x, :, :]], dim=1)
            
        # initialize hidden states
        en_hidden_state, en_cell_state = self.encoder.init_hidden(batch_size=b, image_size=(h, w))
        
        # autoencoder forward
        en_hidden_state, en_cell_state = self.encode(x, seq_len, en_hidden_state, en_cell_state)  # out: [b, h_dim, h, w]    
        
        outputs = self.decode(lag_y, horizon, en_hidden_state, en_cell_state, teacher_forcing_ratio=0.5)
        # outputs = torch.nn.Sigmoid()(outputs)
        # outputs = outputs / 2 + 0.5
        return outputs
    
    def training_step(self, batch, batch_idx):
        
        x, y, _, y_mask = batch
        
        if self.current_epoch > 30:
            out = self(x, y, teacher_forcing_ratio=0.5)
        else:
            out = self(x, y, teacher_forcing_ratio=1.)
            
        loss = self._compute_loss(out[~y_mask], y[~y_mask])
        self.log(f'train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        x, y, _, y_mask = batch
        out = self(x, y, teacher_forcing_ratio=0.)
        loss = self._compute_loss(out[~y_mask], y[~y_mask])
        self.log(f'val_loss', loss)

    def test_step(self, batch, batch_idx):
        
        x, y, _, y_mask = batch
        out = self(x, y, teacher_forcing_ratio=0.)
        loss = self._compute_loss(out[~y_mask], y[~y_mask])
        self.log("test_loss", loss)
