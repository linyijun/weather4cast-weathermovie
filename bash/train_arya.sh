#!/bin/bash -l


python train.py --data_path /data/yijun/WeatherMovie/preprocess-data/ --batch_size 4 --kernel_size 5 --h_dim 32 --seq_len 4 --horizon 4 --source_var_idx 0125 --target_var_idx 0 --model_name convttrans --num_test 10 --gpu_id 1

python test.py --data_path /data/yijun/WeatherMovie/preprocess-data/ --batch_size 4 --kernel_size 3 --h_dim 32 --seq_len 4 --horizon 4 --source_var_idx 0125 --target_var_idx 0 --model_name convttrans_seq4_hoz4_in0125_out0_kernel3_hdim32_1639348116 --num_test 100 --gpu_id 1
