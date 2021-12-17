import os 
import time
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from utils.options import parse_args, load_param_dict
from utils.data_loader import WMDataset, split_train_val_test
from models import seq2seq, unet, convttransformer
from utils.evaluation import rmse_error, r2_error


def test(model, data_loader, num_test, device):
    
    model.eval()
    history, ground_truth, prediction, his_mask, gt_mask = [], [], [], [], []
    
    with torch.no_grad():
        
        for it, data in enumerate(data_loader):
                
            batch_x = data[0].to(device)
            batch_y = data[1].to(device) 
            batch_x_mask = data[2].to(device)
            batch_y_mask = data[3].to(device)
            out = model(batch_x) 
            history.append(batch_x.cpu().data.numpy())
            ground_truth.append(batch_y.cpu().data.numpy())
            prediction.append(out.cpu().data.numpy())
            
            his_mask.append(batch_x_mask.cpu().data.numpy())            
            gt_mask.append(batch_y_mask.cpu().data.numpy())
            
            print(it)
            if it == num_test - 1:
                break

    history = np.concatenate(history)
    ground_truth = np.concatenate(ground_truth)
    prediction = np.concatenate(prediction)
    his_mask = np.concatenate(his_mask)        
    gt_mask = np.concatenate(gt_mask)    
    
    rmse = rmse_error(ground_truth[~gt_mask], prediction[~gt_mask])
    r2 = r2_error(ground_truth[~gt_mask], prediction[~gt_mask])
    print('TEST - RMSE = {:.6f}, R2 = {:.6f}'.format(rmse, r2))
    
    return history, ground_truth, prediction, his_mask, gt_mask

    
def main():
    
    args = parse_args()
    params = load_param_dict(args, mode='test')
    
    device = torch.device('cuda:{}'.format(params['gpu_id']) if torch.cuda.is_available() else 'cpu')
    logging.info(device)

    dataset = WMDataset(data_path=params['data_path'], 
                        sample_path=params['sample_path'],
                        region_id=params['region_id'], 
                        source_vars=params['source_vars'],
                        target_vars=params['target_vars'],
                        seq_len=params['seq_len'],
                        horizon=params['horizon'],
                        use_static=params['use_static'])
    
    _, _, test_dataset = split_train_val_test(dataset)
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
    if 'seq2seq' in params['model_name']:
        model = seq2seq.EncoderDecoderConvLSTM(in_channels=len(dataset.source_vars),
                                               h_channels=params['h_dim'], 
                                               out_channels=len(dataset.target_vars),
                                               kernel_size=(params['kernel_size'], params['kernel_size'])).to(device)
    elif 'unet' in params['model_name']:
        model = unet.UNet(in_channels=len(dataset.source_vars) * params['seq_len'],
                          out_channels=len(dataset.target_vars) * params['horizon'],).to(device)
    
    elif 'convttrans' in params['model_name']:
        model = convttransformer.ConvTTransformer(in_channels=len(dataset.source_vars),
                                                  embed_size=params['h_dim'],
                                                  out_channels=len(dataset.target_vars),
                                                  num_layers=3,
                                                  num_heads=2,
                                                  kernel_size=(params['kernel_size'], params['kernel_size'])).to(device)
    
    else:
        raise NotImplementedError

        
    model.load_state_dict(torch.load(params['model_path'], map_location=device)['state_dict'], strict=False)
    history, ground_truth, prediction, his_mask, gt_mask = test(model, test_loader, num_test=params['num_test'], device=device)
    
    np.savez_compressed(
        os.path.join('{}/prediction.npz'.format(params['result_path'])),
        history=history,
        ground_truth=ground_truth,
        prediction=prediction,
        history_mask=his_mask,
        ground_truth_mask=gt_mask,)

    
if __name__ == "__main__":
    main()
