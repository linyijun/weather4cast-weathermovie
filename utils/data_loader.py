import time
import os
import h5py
import csv
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Subset
from torch.utils.data import Dataset


class WMDataset(Dataset):
    
    def __init__(self, data_path, sample_path, 
                 region_id='R1', 
                 source_vars=['temperature', 'crr_intensity', 'asii_turb_trop_prob', 'cma'], 
                 target_vars=['temperature', 'crr_intensity', 'asii_turb_trop_prob', 'cma'], 
                 seq_len=4, 
                 horizon=1, 
                 use_static=False, **kwargs):
        """
        Params:
            data_path (str): path to the parent folder containing .h5 
            sample_path (str): path to the file contain samples 
            region_id (str): region to load data from. Default: 'R1'.
            source_vars (list): input sequence variables
            target_vars (list): output sequence variables, by default same as source variables
            seq_len (int): input sequence length. Default: 4
            horizon (int): output sequence length. Default: 4.
            use_static (boolean): use static features. Default: False.
        """
    
        # data dimensions
        self.data_path = data_path
        self.region_id = region_id
        self.source_vars = source_vars
        self.target_vars = target_vars
        self.seq_len = seq_len
        self.horizon = horizon
        self.use_static = use_static
        
        # load static variables if any 
        static_path = os.path.join(data_path, 'static', f'static_{region_id}.h5')
        hf = h5py.File(static_path, 'r')
        self.static_data = np.array(hf['data'])
        
        # load sample indices, 
        # [day_in_year, start_bin_id, end_bin_id, next_day_in_year, next_start_bin_id, next_end_bin_id]
        self.samples = pd.read_csv(sample_path, header=0)[:1000]
    
    def __len__(self):
        """ total number of samples (sequences of in:4-out:32 in our case) to train """
        return len(self.samples)

    def load_sequence_netcdf4(self, day_in_year, start_bin_id, end_bin_id):
        """ load one sequence from netcdf4 """
        
        path = os.path.join(self.data_path, self.region_id, str(day_in_year) + '.h5')
        assert os.path.exists(path), "Input path does not exist."
                            
        hf = h5py.File(path, 'r')
        value = np.array(hf['data']['value'])
        mask = np.array(hf['data']['mask'])
        value = value[start_bin_id: end_bin_id + 1, ...]
        mask = mask[start_bin_id: end_bin_id + 1, ...]        
        descriptions = list(hf['data'].attrs['descriptions'])
        return value, mask, descriptions
     
    def __getitem__(self, idx):
        """ load one sample """
        
        sample = self.samples.iloc[idx]
        day_in_year = int(sample['day_in_year'])
        start_bin_id = int(sample['start_bin_id'])
        end_bin_id = int(sample['end_bin_id'])
        next_day_in_year = int(sample['next_day_in_year']) if not pd.isna(sample['next_day_in_year']) else None
        next_start_bin_id = int(sample['next_start_bin_id']) if not pd.isna(sample['next_start_bin_id']) else None
        next_end_bin_id = int(sample['next_end_bin_id']) if not pd.isna(sample['next_end_bin_id']) else None

        seq, mask, desc = self.load_sequence_netcdf4(day_in_year, start_bin_id, end_bin_id)
        if next_day_in_year is not None:
            next_seq, next_mask, _ = self.load_sequence_netcdf4(next_day_in_year, next_start_bin_id, next_end_bin_id)
            seq = np.concatenate([seq, next_seq])
            mask = np.concatenate([mask, next_mask])            
        
        source_index = [desc.index(v) for v in self.source_vars]
        target_index = [desc.index(v) for v in self.target_vars]
        in_seq = seq[:self.seq_len, source_index, ...]
        out_seq = seq[self.seq_len:(self.seq_len + self.horizon), target_index, ...]
        in_mask = mask[:self.seq_len, source_index, ...]
        out_mask = mask[self.seq_len:(self.seq_len + self.horizon), target_index, ...]
        in_seq = in_seq.astype(np.float32)
        out_seq = out_seq.astype(np.float32)
        return in_seq, out_seq, in_mask, out_mask
        
    def get_item(self, idx=0):
        """ this function load one sample for debugging """
        
        in_seq, out_seq, in_mask, out_mask = self.__getitem__(idx)
        in_seq = np.expand_dims(in_seq, axis=0)
        out_seq = np.expand_dims(out_seq, axis=0)
        in_mask = np.expand_dims(in_mask, axis=0)        
        out_mask = np.expand_dims(out_mask, axis=0)
        return in_seq, out_seq, in_mask, out_mask
    

def split_train_val_test(dataset, num_test=None):
    
    idx = [i for i in range(len(dataset))]
    
    random.seed(1234)
    random.shuffle(idx)
    
    num_train = int(0.8 * int(0.8 * len(idx)))
    num_val = int(0.2 * int(0.8 * len(idx)))
    
    train_idx = idx[:num_train]
    val_idx = idx[num_train: (num_train + num_val)]
    
    if num_test is None:
        test_idx = idx[(num_train + num_val):]
    else:
        test_idx = idx[(num_train + num_val):][:num_test]

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    return train_dataset, val_dataset, test_dataset

