import sys
import time
import argparse
import os
import logging
import glob


def parse_args():
    """ this class includes training options """
    
    # basic parameters
    parser = argparse.ArgumentParser(description='weather4cast')
    
    # load data from file
    parser.add_argument('--data_path', type=str, default='/home/yaoyi/lin00786/weather4cast/preprocess-data/',
                        help='data path, dynamic variables')
    parser.add_argument('--sample_path', type=str, default='./samples.csv', 
                        help='data path, splitting information')
    parser.add_argument('--region_id', type=str, default='R1', 
                        help="region_id to load data from. Default: R1")
    
    parser.add_argument('--model_name', type=str, default='test', help='model name')
    parser.add_argument('--result_path', type=str, default='./results',
                         help='result path, including log file and model file')
    
    parser.add_argument('--source_var_idx', type=str, default='0', help='Default: temperature')
    parser.add_argument('--target_var_idx', type=str, default='0', help='Default: temperature')
    
    # training parameters
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')
    parser.add_argument('--num_epochs', type=int, default=600, help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='the scalar for l2 loss')
    parser.add_argument('--patience', type=int, default=10, help='the patience for early stop')
    parser.add_argument('--log_interval', type=int, default=1, help='the interval to test model')
    parser.add_argument('--h_dim', type=int, default=64, help='size of hidden')
    parser.add_argument('--kernel_size', type=int, default=3, help='size of kernel')

    # model parameters
    parser.add_argument('--seq_len', type=int, default=4, help='input sequence length')
    parser.add_argument('--horizon', type=int, default=1, help='output sequence length')
    parser.add_argument('--use_static', action='store_true', help='use static features')
    
    # others
    parser.add_argument('--verbose', action='store_false', help='print more debugging information. Default: True')
    parser.add_argument('--num_test', type=int, default=None, help='the number of testing samples during inference')

    args = parser.parse_args()
    return args


def verbose(params):

    if params['verbose']:
        logging.basicConfig(filename=params['log_file_path'], 
                            level=logging.INFO, 
                            format='%(asctime)s - %(message)s',
                            datefmt='%y-%m-%d %H:%M')
        logging.getLogger().addHandler(logging.StreamHandler())

        for k, v in params.items():
            logging.info(f'{k} - {v}')


def input_check(params):

    if not os.path.exists(params['data_path']):
        print('Data path {} does not exist.'.format(params['data_path']))
        sys.exit(-1)
 
    if not os.path.exists(params['sample_path']):
        print('Sample path {} does not exist.'.format(params['sample_path']))
        sys.exit(-1)
        
    if params['mode'] == 'train':
        if not os.path.exists(params['result_path']):
            os.makedirs(params['result_path'])
            os.makedirs(params['model_path'])
            os.makedirs(params['log_path'])
        
    elif params['mode'] == 'test':
        if not os.path.exists(params['model_path']):
            print('Model path {} does not exist.'.format(params['model_path']))
            sys.exit(-1)
        
        model_path = '{}/*.ckpt'.format(params['model_path'])
        model_path = glob.glob(model_path)
        assert len(model_path)==1, f'All these files were found: {model_path}'
        params['model_path'] = model_path[0]
            
    return params

            
    
def load_param_dict(args=None, mode='train'):
    
    param_dict = dict()
    param_dict['mode'] = mode
    param_dict['vars'] = ['temperature', 'ctth_tempe', 'ishai_skt', 'crr_intensity', 'asii_turb_trop_prob', 'cma']
    
    if args is None:
        param_dict['data_path'] = '/home/yaoyi/lin00786/weather4cast/preprocess-data/'
        param_dict['sample_path'] = './samples.csv'
        param_dict['region_id'] = 'R1'
        param_dict['model_name'] = 'Seq2Seq_seq4_hoz1_in1_out1_kernel1'
        param_dict['result_path'] = '/home/yaoyi/lin00786/weather4cast/weather4cast-test-lightning/results'
        param_dict['source_var_idx'] = [1]
        param_dict['target_var_idx'] = [1]
                   
        param_dict['gpu_id'] = 0
        param_dict['num_epochs'] = 100
        param_dict['batch_size'] = 32
        param_dict['lr'] = 0.001
        param_dict['weight_decay'] = 0.001
        param_dict['patience'] = 10
        param_dict['log_interval'] = 1
        param_dict['h_dim'] = 32
        param_dict['kernel_size'] = 3
        
        param_dict['seq_len'] = 4
        param_dict['horizon'] = 1
        param_dict['use_static'] = False
        
        param_dict['num_test'] = 1
        param_dict['verbose'] = True

    else:
        param_dict['data_path'] = args.data_path
        param_dict['sample_path'] = args.sample_path
        param_dict['region_id'] = args.region_id
        param_dict['model_name'] = args.model_name
        param_dict['result_path'] = args.result_path
        param_dict['source_var_idx'] = [int(i) for i in list(args.source_var_idx)]
        param_dict['target_var_idx'] = [int(i) for i in list(args.target_var_idx)]
        
        param_dict['gpu_id'] = args.gpu_id
        param_dict['num_epochs'] = args.num_epochs
        param_dict['batch_size'] = args.batch_size
        param_dict['lr'] = args.lr
        param_dict['weight_decay'] = args.weight_decay
        param_dict['patience'] = args.patience
        param_dict['log_interval'] = args.log_interval
        param_dict['h_dim'] = args.h_dim
        param_dict['kernel_size'] = args.kernel_size
        
        param_dict['seq_len'] = args.seq_len
        param_dict['horizon'] = args.horizon
        param_dict['use_static'] = args.use_static
        
        param_dict['num_test'] = args.num_test
        param_dict['verbose'] = args.verbose
    
    param_dict['source_vars'] = [param_dict['vars'][i] for i in param_dict['source_var_idx']]
    param_dict['target_vars'] = [param_dict['vars'][i] for i in param_dict['target_var_idx']]
    
    if mode == 'train':
        model_name = ''
        if param_dict['model_name'] == 'seq2seq':
            model_name = '{}_seq{}_hoz{}_in{}_out{}_kernel{}_hdim{}'.format(param_dict['model_name'],
                                                                            param_dict['seq_len'],
                                                                            param_dict['horizon'],
                                                                            ''.join([str(i) for i in param_dict['source_var_idx']]),
                                                                            ''.join([str(i) for i in param_dict['target_var_idx']]),
                                                                            param_dict['kernel_size'],
                                                                            param_dict['h_dim'],)
        if param_dict['model_name'] == 'unet':
            model_name = '{}_seq{}_hoz{}_in{}_out{}'.format(param_dict['model_name'],
                                                            param_dict['seq_len'],
                                                            param_dict['horizon'],
                                                            ''.join([str(i) for i in param_dict['source_var_idx']]),
                                                            ''.join([str(i) for i in param_dict['target_var_idx']]),)

        if param_dict['model_name'] == 'convttrans':
            model_name = '{}_seq{}_hoz{}_in{}_out{}_kernel{}_hdim{}'.format(param_dict['model_name'],
                                                                            param_dict['seq_len'],
                                                                            param_dict['horizon'],
                                                                            ''.join([str(i) for i in param_dict['source_var_idx']]),
                                                                            ''.join([str(i) for i in param_dict['target_var_idx']]),
                                                                            param_dict['kernel_size'],
                                                                            param_dict['h_dim'],)

        
        param_dict['result_path'] = os.path.join(param_dict['result_path'], 
                                                 model_name + '_' + str(int(time.time())))        
    else:
        model_name = args.model_name
        param_dict['result_path'] = os.path.join(param_dict['result_path'], model_name)        
        
    param_dict['model_path'] = os.path.join(param_dict['result_path'], 'models')
    param_dict['log_path'] = os.path.join(param_dict['result_path'], 'logs')
    param_dict['log_file_path'] = os.path.join(param_dict['log_path'], model_name + f'_{mode}.log')

    print(param_dict.keys())
    param_dict = input_check(param_dict)
    verbose(param_dict)
    return param_dict