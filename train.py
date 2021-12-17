import os 
import json
import time
import logging
import math
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from models import seq2seq, unet, convttransformer
from utils.options import parse_args, load_param_dict
from utils.data_loader import WMDataset, split_train_val_test


def train(params):
    
    device = params['gpu_id'] if torch.cuda.is_available() else None
    logging.info(f'Device - GPU:{device}')

    dataset = WMDataset(data_path=params['data_path'], 
                        sample_path=params['sample_path'],
                        region_id=params['region_id'], 
                        source_vars=params['source_vars'],
                        target_vars=params['target_vars'],
                        seq_len=params['seq_len'],
                        horizon=params['horizon'],
                        use_static=params['use_static'])
    
    train_dataset, val_dataset, test_dataset = split_train_val_test(dataset)
    logging.info('#Train={}, #Val={}, #Test={}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))
    logging.info('Source Variables - {}'.format(dataset.source_vars))
    logging.info('Target Variables - {}'.format(dataset.target_vars))
        
    train_loader = DataLoader(
        train_dataset, 
        batch_size=params['batch_size'], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=params['batch_size'], 
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False)
    
    if params['model_name'] == 'seq2seq':
        model = seq2seq.EncoderDecoderConvLSTM(in_channels=len(dataset.source_vars),
                                               h_channels=params['h_dim'], 
                                               out_channels=len(dataset.target_vars),
                                               kernel_size=(params['kernel_size'], params['kernel_size']))
    elif params['model_name'] == 'unet':
        model = unet.UNet(in_channels=len(dataset.source_vars) * params['seq_len'],
                          out_channels=len(dataset.target_vars) * params['horizon'],)
    
    elif params['model_name'] == 'convttrans':
        model = convttransformer.ConvTTransformer(in_channels=len(dataset.source_vars),
                                                  embed_size=params['h_dim'],
                                                  out_channels=len(dataset.target_vars),
                                                  num_layers=3,
                                                  num_heads=2,
                                                  kernel_size=(params['kernel_size'], params['kernel_size']))
    
    else:
        raise NotImplementedError

    tensorboard_logger = TensorBoardLogger(
        save_dir=params['log_path'],
        name='tensorboard_log',
    )

    csv_logger = CSVLogger(
        save_dir=params['log_path'], 
        name='csv_log', 
    )
     
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath=params['model_path'],
        filename='{epoch}-{val_loss:.5f}',
    )
    
    earlystop_callback = EarlyStopping(
        monitor='val_loss', 
        min_delta=0.00, 
        mode='min',
        patience=params['patience'], 
        verbose=False, 
    )
    
    trainer = pl.Trainer(
        max_epochs=params['num_epochs'],
        gpus= [device],
        log_every_n_steps=10,
        progress_bar_refresh_rate=0.5,
        logger=[tensorboard_logger, csv_logger],
        callbacks=[checkpoint_callback, earlystop_callback],
        # limit_train_batches=0.2,
    )
    
    trainer.fit(model, train_loader, val_loader)

    result_val = trainer.test(dataloaders=test_loader, ckpt_path='best')
    
    output_json = {
        'model_name': params['model_name'],
        'seq_len': params['seq_len'],
        'horizon': params['horizon'],
        'test_loss': result_val[0]['test_loss'],
        'best_model_path': checkpoint_callback.best_model_path,
    }

    output_json_path = os.path.join(params['result_path'], 'output.json')
    with open(output_json_path, "w") as f:
        json.dump(output_json, f, indent=4)

    return output_json

    
def main():
    
    args = parse_args()
    params = load_param_dict(args, mode='train')
    train(params)
    
    
if __name__ == "__main__":
    main()
