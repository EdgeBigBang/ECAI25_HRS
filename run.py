import argparse
import random

import numpy as np
import torch

from exp import Exp_Forecast

if __name__ == '__main__':
    fix_seed = 2021
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    random.seed(fix_seed)
    torch.cuda.manual_seed_all(fix_seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description='Multimodal TSF')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='1 for train or 0 for test')
    parser.add_argument('--draw_test', type=int, default=0, help='draw test result')
    parser.add_argument('--task_id', type=str, default='test', help='task id')
    parser.add_argument('--model', type=str, default='HRS', help='model name')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--data', type=str, default='Cloud', help='data type')
    parser.add_argument('--dir_path', type=str, default='./data/', help='dir path')
    parser.add_argument('--data_path', type=str, default='ECW_09.csv', help='data file name')
    parser.add_argument('--features', type=str, default='S',
                            help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--n_feature', type=int, default=1, help='features number, if CI, set 1')
    parser.add_argument('--target', type=str, default='cnt', help='target feature in S or MS task')
    parser.add_argument('--seq_len', type=int, default=24, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=12, help='start token length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    parser.add_argument('--inverse', type=int, default=1, help='inverse output data')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]]')
    
    # scheduling aware loss set
    parser.add_argument('--scheduling_aware', type=int, default=1, help='1 for scheduling aware loss, 0 for mse loss')

    # image-based representation config
    parser.add_argument('--h', type=int, default=24, help='h')
    parser.add_argument('--maxScal', type=float, default=2.79, help='max scal')
    parser.add_argument('--expand', type=int, default=1, help='expand ratio')
    parser.add_argument('--lw', type=float, default=1, help='line width')
    parser.add_argument('--lc', type=float, nargs='+', default=(0, 0, 0), help='line color')
    parser.add_argument('--bc', type=float, nargs='+', default=(1, 1, 1), help='background color')
    parser.add_argument('--channel', type=int, default=3, help='3 for RGB and 1 for Grey')
    parser.add_argument('--hidden_dim', type=int, default=8, help='hidden dimension')  ## 原本是8
    parser.add_argument('--patch_size', type=int, nargs='+', default=(4, 4), help='patch_size') #这是卷积核的大小
    parser.add_argument('--stride', type=int, nargs='+', default=(2, 2), help='stride') # 与上面重叠一半
    parser.add_argument('--token_mlp_dim', type=int, default=20, help='token_mlp_dim')
    parser.add_argument('--dimension_mlp_dim', type=int, default=8, help='dimension_mlp_dim')
    parser.add_argument('--n_blocks', type=int, default=4, help='h')

    # model define
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--ffm_len', type=int, default=1024, help='ffm_len')


    # optimization
    parser.add_argument('--num_workers', type=int, default=20, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=60, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stop')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='1', help='gpu ids of multiple gpus')

    # Transformer-based
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    
    
    args = parser.parse_args()
    args.model_type = 0 if args.model == 'HRS' else 1
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    print('Args: {}'.format(args))

    exp = Exp_Forecast(args)
    if args.is_training:
        setting = '{}_h{}_lw{}_exp{}_ch{}_hd{}_cmd{}_ps{}_{}_ss{}_{}_tmd{}_nb{}_dr{}_lr{}'.format(
            args.task_id,
            args.h,
            args.lw,
            args.expand,
            args.channel,
            args.hidden_dim,
            args.dimension_mlp_dim,
            args.patch_size[0],
            args.patch_size[1],
            args.stride[0],
            args.stride[1],
            args.token_mlp_dim,
            args.n_blocks,
            args.dropout,
            args.learning_rate
        )

        print('>>>>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>>>>start testing : {}>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.test(setting)
    else:
        setting = '{}_h{}_lw{}_exp{}_ch{}_hd{}_cmd{}_ps{}_{}_ss{}_{}_tmd{}_nb{}_dr{}_lr{}'.format(
            args.task_id,
            args.h,
            args.lw,
            args.expand,
            args.channel,
            args.hidden_dim,
            args.dimension_mlp_dim,
            args.patch_size[0],
            args.patch_size[1],
            args.stride[0],
            args.stride[1],
            args.token_mlp_dim,
            args.n_blocks,
            args.dropout,
            args.learning_rate
        )

        print('>>>>>>>>>>start testing : {}>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.test(setting, test=1)

    input('Press Enter to continue...')

