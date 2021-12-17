import torch
import torch.nn as nn
# from One_hot_encoder import One_hot_encoder
from models.model_base import ModelBase


class TSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        
        super(TSelfAttention, self).__init__()
        
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        self.fc = nn.Linear(self.num_heads * self.head_dim, embed_size)

    def forward(self, value, key, query):
                
        B, H, W, T, C = query.size()
        
        # Split the embedding into self.heads different pieces
        value = value.reshape(B, H, W, T, self.num_heads, self.head_dim)  # embed_size维拆成 heads×head_dim
        key   = key.reshape(B, H, W, T, self.num_heads, self.head_dim)
        query = query.reshape(B, H, W, T, self.num_heads, self.head_dim)

        value = self.values(value)  # [B, H, W, T, num_heads, head_dim]
        key   = self.keys(key)      # 
        query = self.queries(query) # 

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm
        
        energy = torch.einsum("bhwqnd,bhwknd->bhwnqk", [query, key])   # self-attention
        # query: [B, H, W, T, num_heads, head_dim]
        # key: [B, H, W, T, num_heads, head_dim]
        # energy: [B, H, W, num_heads, T, T]
        
        # Normalize energy values
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=-1)  # 在K维做softmax，和为1
        # attention: [B, H, W, num_heads, T, T]
        
        out = torch.einsum("bhwnqk,bhwvnd->bhwvnd", [attention, value]).reshape(
                B, H, W, T, self.num_heads * self.head_dim)
        # attention: [B, H, W, num_heads, T, T]
        # values: [B, H, W, T, num_heads, head_dim]
        # out: [B, H, W, T, num_heads, head_dim], then [B, H, W, T, num_heads x head_dim]
        
        out = self.fc(out)  # (B, H, W, T, embed_size)
        return out
    
    
class TTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        
        super(TTransformerEncoderLayer, self).__init__()
        
        # temporal embedding
        self.pos_embedding = nn.Embedding(16, embedding_dim=embed_size)
        
        self.attn = TSelfAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.ff = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        B, T, C, H, W = x.size()        
    
        pos_encoder = (
            torch.arange(0, T, device=x.device)
            .unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            .repeat(B, 1, H, W)
        )  # pos_encoder: [B, T, H, W]
        
        pos_encoder = self.pos_embedding(pos_encoder)  # pos_encoder: [B, T, H, W, C]
        pos_encoder = pos_encoder.permute(0, 1, 4, 2, 3)
        assert (
            pos_encoder.size(-1) == C, 
            "positional encoder channel does not match"
        )
        
        # adding temporal embedding to query
        x = x + pos_encoder  # [B, T, C, H, W]
        x = x.permute(0, 3, 4, 1, 2)
        value, key, query = x, x, x
        
        attention = self.attn(value, key, query)

        # add skip connection, run through normalization and finally dropout        
        x = self.dropout(self.norm1(attention + query))
        out = self.ff(x)
        out = self.dropout(self.norm2(out + x))  # (B, H, W, T, embed_size)
        out = out.permute(0, 3, 4, 1, 2)
        return out  # (B, T, embed_size, H, W)


class ConvTTransformerBlock(nn.Module):
    def __init__(
        self, 
        embed_size, 
        num_heads,
        kernel_size,
        dropout, 
        forward_expansion,
    ):

        super(ConvTTransformerBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=embed_size,
                      out_channels=embed_size * forward_expansion,
                      kernel_size=kernel_size,
                      padding=0,
                      bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=embed_size * forward_expansion,
                      out_channels=embed_size,
                      kernel_size=(1, 1),
                      padding=0,
                      bias=True),)

        self.conv_trans = nn.ConvTranspose2d(in_channels=embed_size,
                                             out_channels=embed_size,
                                             kernel_size=kernel_size,
                                             padding=0,
                                             bias=True)
        
        self.TTransformer = TTransformerEncoderLayer(embed_size, num_heads, dropout, forward_expansion)
    
    def forward(self, x):
        # x: [B, T, C, H, W]
        
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        conv_x = self.conv(x)  # [B * T, C', H', W']
                
        _, C, H, W =  conv_x.size()
        conv_x = conv_x.view(B, T, C, H, W)
        
        out = self.TTransformer(conv_x) + conv_x  # (B, T, C', H', W')
        
        out = out.reshape(B * T, C, H, W)
        out = self.conv_trans(out)
        _, C, H, W = out.size()
        out = out.view(B, T, C, H, W)
        return out


class ConvTTransformer(ModelBase):
    def __init__(
        self, 
        in_channels = 4, 
        out_channels = 1,
        seq_len = 4,
        horizon = 4,
        embed_size = 16, 
        num_layers = 3,
        num_heads = 2,
        kernel_size=(3,3),
        forward_expansion=4,
        lr: float=1e-4,
        weight_decay: float=1e-4
    ):        
        super().__init__(lr=lr,
                         weight_decay=weight_decay)
        
        self.save_hyperparameters()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=embed_size, 
                               kernel_size=(1, 1),
                               padding=(0, 0))
        
        self.layers = nn.ModuleList(
            [
                ConvTTransformerBlock(
                    embed_size=embed_size,
                    num_heads=num_heads,
                    dropout=0.1,
                    kernel_size=kernel_size,
                    forward_expansion=forward_expansion)
                for _ in range(num_layers)
            ]
        )
        
                
        # 缩小时间维度。  例：T_dim=12到output_T_dim=3，输入12维降到输出3维
        self.conv2 = nn.Conv2d(in_channels=seq_len, 
                               out_channels=horizon, 
                               kernel_size=(1, 1),
                               padding=(0, 0))
        
        # 缩小通道数，降到1维。
        self.conv3 = nn.Conv2d(in_channels=embed_size, 
                               out_channels=out_channels, 
                               kernel_size=(1, 1),
                               padding=(0, 0))
    
    def forward(self, src):
        """
        params:
            x (5-D Tensor) - [B, T, C, H, W]  #   batch, seq_len, channel, height, width
        """

        B, T, C, H, W = src.size()
        
        src_trans = src.view(B * T, C, H, W)
        src_trans = self.conv1(src_trans)  # x: [B * T, C', H, W]
        src_trans = src_trans.view(B, T, self.hparams.embed_size, H, W)
        
        out = src_trans
        for layer in self.layers:
            out = layer(out)  # out: [B, T, C, H, W]  
        
        B, T, C, H, W = out.size()
        out = out.permute(0, 2, 1, 3, 4)  # out: [B, C, T, H, W]
        out = out.reshape(B * C, T, H, W)
        out = self.conv2(out)
        out = out.reshape(B, C, -1, H, W)
        
        B, C, T, H, W = out.size()
        out = out.permute(0, 2, 1, 3, 4)  # out: [B, T, C, H, W]  
        out = out.reshape(B * T, C, H, W)
        out = self.conv3(out)        
        out = out.reshape(B, T, -1, H, W)
        
        return out
        
    def training_step(self, batch, batch_idx):
        
        x, y, _, y_mask = batch
        
        if self.current_epoch > 30:
            out = self(x)
        else:
            out = self(x)
            
        loss = self._compute_loss(out[~y_mask], y[~y_mask])
        self.log(f'train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        x, y, _, y_mask = batch
        out = self(x)
        loss = self._compute_loss(out[~y_mask], y[~y_mask])
        self.log(f'val_loss', loss)

    def test_step(self, batch, batch_idx):
        
        x, y, _, y_mask = batch
        out = self(x)
        loss = self._compute_loss(out[~y_mask], y[~y_mask])
        self.log("test_loss", loss)
