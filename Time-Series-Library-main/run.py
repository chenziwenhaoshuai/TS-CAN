import argparse
import os
import torch
import torch.backends
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--huber_delta', type=float, default=1.0, help='SmoothL1/Huber loss beta')
    parser.add_argument('--loss_mse_weight', type=float, default=0.5, help='MSE weight for MSEMAE loss')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'])
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--weight_averaging', type=str, default='none', choices=['none', 'ema'])
    parser.add_argument('--ema_decay', type=float, default=0.995)
    parser.add_argument('--ema_start_epoch', type=int, default=1)
    parser.add_argument('--vali_metric_mode', type=str, default='all', choices=['all', 'tail', 'weighted'])
    parser.add_argument('--vali_metric_horizon_start', type=int, default=0)
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument(
        '--test_every_epoch',
        type=int,
        default=1,
        help='evaluate full test metrics after every epoch and save epoch_test_metrics.csv',
    )

    # GPU
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu (default: on)')
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu', help='disable gpu (force cpu)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', action='store_true', default=False,
                        help='enable dtw metric (time consuming; default: off)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--can_stride', type=int, default=8, help='patch stride for CANPatchTST')
    parser.add_argument('--can_shifts', type=str, default='1,2,4,8',
                        help='comma separated sparse shift list for CAN interactions')
    parser.add_argument('--can_temporal_shifts', type=str, default='')
    parser.add_argument('--can_cli_mode', type=str, default='full', choices=['full', 'inner', 'wedge', 'adaptive'],
                        help='Clifford interaction mode for CANPatchTST')
    parser.add_argument('--can_temporal_cli_mode', type=str, default='inner',
                        choices=['full', 'inner', 'wedge', 'adaptive'],
                        help='temporal Clifford interaction mode for CANPatchTST')
    parser.add_argument('--can_ctx_mode', type=str, default='diff', choices=['diff', 'abs'],
                        help='context mode for CANPatchTST')
    parser.add_argument('--can_drop_path', type=float, default=0.1, help='drop path rate for CAN blocks')
    parser.add_argument('--can_drop_path_schedule', type=str, default='linear', choices=['linear', 'uniform'])
    parser.add_argument('--can_kernel_size', type=int, default=3, help='depthwise kernel size in CAN local context')
    parser.add_argument('--can_init_values', type=float, default=1e-5, help='LayerScale init value for CAN blocks')
    parser.add_argument('--can_gamma_lr_scale', type=float, default=1.0)
    parser.add_argument('--can_gamma_weight_decay', type=float, default=0.0)
    parser.add_argument('--can_use_gffng', type=int, default=1, help='use global geometric context branch')
    parser.add_argument('--can_global_cli_mode', type=str, default='inner',
                        choices=['full', 'inner', 'wedge', 'adaptive'])
    parser.add_argument('--can_global_ctx_mode', type=str, default='abs', choices=['diff', 'abs'])
    parser.add_argument('--can_global_shifts', type=str, default='')
    parser.add_argument('--can_temporal_roll', type=int, default=1, help='use temporal sparse rolling branch')
    parser.add_argument('--can_temporal_circular', type=int, default=0)
    parser.add_argument('--can_beta_init', type=float, default=0.5, help='initial beta for extra CAN branches')
    parser.add_argument('--can_temporal_beta_init', type=float, default=None)
    parser.add_argument('--can_global_beta_init', type=float, default=None)
    parser.add_argument('--can_use_orth', type=int, default=0, help='use context orthogonalization in CAN variants')
    parser.add_argument('--can_context_pyramid', type=int, default=0,
                        help='use multi-scale context pyramid in CANPatchTST')
    parser.add_argument('--can_use_ffn', type=int, default=0)
    parser.add_argument('--can_cross_var', type=int, default=0)
    parser.add_argument('--can_cross_var_layers', type=int, default=1)
    parser.add_argument('--can_cross_var_context', type=str, default='others_mean',
                        choices=['others_mean', 'mean'])
    parser.add_argument('--can_cross_var_shifts', type=str, default='1,2,4,8,16')
    parser.add_argument('--can_var_attn', type=int, default=0)
    parser.add_argument('--can_var_attn_layers', type=int, default=1)
    parser.add_argument('--can_var_attn_dim', type=int, default=32)
    parser.add_argument('--can_var_attn_top_k', type=int, default=0)
    parser.add_argument('--can_var_attn_shifts', type=str, default='1,2,4,8')
    parser.add_argument('--can_var_embed', type=int, default=0)
    parser.add_argument('--can_time_mark', type=int, default=0)
    parser.add_argument('--can_time_mark_mode', type=str, default='flatten',
                        choices=['flatten', 'last'])
    parser.add_argument('--can_time_mark_scale_init', type=float, default=1.0)
    parser.add_argument('--can_linear_residual', type=int, default=0)
    parser.add_argument('--can_linear_mode', type=str, default='raw', choices=['raw', 'decomp'])
    parser.add_argument('--can_linear_individual', type=int, default=0)
    parser.add_argument('--can_linear_scale_init', type=float, default=0.5)
    parser.add_argument('--can_periodic_residual', type=int, default=0)
    parser.add_argument('--can_periods', type=str, default='24')
    parser.add_argument('--can_periodic_alpha', type=float, default=0.2)
    parser.add_argument('--can_periodic_learnable', type=int, default=0)
    parser.add_argument('--can_coarse_var_attn', type=int, default=0)
    parser.add_argument('--can_coarse_var_levels', type=int, default=3)
    parser.add_argument('--can_coarse_var_dim', type=int, default=32)
    parser.add_argument('--can_coarse_var_scale_init', type=float, default=0.1)
    parser.add_argument('--can_coarse_var_mode', type=str, default='diff',
                        choices=['diff', 'abs'])
    parser.add_argument('--can_hierarchical_mixer', type=int, default=0)
    parser.add_argument('--can_hierarchical_levels', type=int, default=3)
    parser.add_argument('--can_hierarchical_layers', type=int, default=1)
    parser.add_argument('--can_hierarchical_dim', type=int, default=64)
    parser.add_argument('--can_hierarchical_cross_scale_init', type=float, default=0.05)
    parser.add_argument('--can_hierarchical_fusion_init', type=float, default=0.2)
    parser.add_argument('--can_hierarchical_mode', type=str, default='blend',
                        choices=['blend', 'residual'])
    parser.add_argument('--can_hierarchical_residual_scale_init', type=float, default=1.0)
    parser.add_argument('--can_periodic_image', type=int, default=0)
    parser.add_argument('--can_periodic_image_top_k', type=int, default=3)
    parser.add_argument('--can_periodic_image_dim', type=int, default=32)
    parser.add_argument('--can_periodic_image_layers', type=int, default=1)
    parser.add_argument('--can_periodic_image_shifts', type=str, default='1,2,4')
    parser.add_argument('--can_periodic_image_scale_init', type=float, default=0.0)
    parser.add_argument('--can_deep_periodic_image', type=int, default=0)
    parser.add_argument('--can_deep_periodic_top_k', type=int, default=3)
    parser.add_argument('--can_deep_periodic_layers', type=int, default=1)
    parser.add_argument('--can_deep_periodic_shifts', type=str, default='1,2,4')
    parser.add_argument('--can_deep_periodic_scale_init', type=float, default=0.1)
    parser.add_argument('--can_multiscale_patch_lens', type=str, default='')
    parser.add_argument('--can_multiscale_stride_ratio', type=float, default=0.5)
    parser.add_argument('--can_multiscale_main_bias', type=float, default=0.0)

    # GCN
    parser.add_argument('--node_dim', type=int, default=10, help='each node embbed to dim dimentions')
    parser.add_argument('--gcn_depth', type=int, default=2, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=32, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')

    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # TimeFilter
    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding. Set pos to 0 or 1')

    args = parser.parse_args()
    # Use user-provided seed for all stochastic components.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)


    if args.task_name == 'long_term_forecast':
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        from exp.exp_imputation import Exp_Imputation
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        from exp.exp_anomaly_detection import Exp_Anomaly_Detection
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        from exp.exp_classification import Exp_Classification
        Exp = Exp_Classification
    elif args.task_name == 'zero_shot_forecast':
        from exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast
        Exp = Exp_Zero_Shot_Forecast
    else:
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.use_gpu:
                if args.gpu_type == 'mps':
                    torch.backends.mps.empty_cache()
                elif args.gpu_type == 'cuda':
                    torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.use_gpu:
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
