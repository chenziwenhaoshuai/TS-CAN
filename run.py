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
    parser.add_argument('--results', type=str, default='./results/', help='location of prediction results')
    parser.add_argument('--test_results', type=str, default='./test_results/', help='location of test visualizations')

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
    parser.add_argument('--anomaly_score_mode', type=str, default='mean',
                        choices=['mean', 'max', 'topk_mean'],
                        help='aggregate per-variable reconstruction errors into anomaly scores')
    parser.add_argument('--anomaly_score_top_k', type=int, default=3,
                        help='number of largest variable errors used by topk_mean anomaly scoring')
    parser.add_argument('--anomaly_train_stride', type=int, default=0,
                        help='window stride for anomaly train/val loaders; 0 keeps the dataset default')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--tv_dt', type=int, default=0, help='whether to use time variant dt for MambaSL')
    parser.add_argument('--tv_B', type=int, default=0, help='whether to use time variant B for MambaSL')
    parser.add_argument('--tv_C', type=int, default=0, help='whether to use time variant C for MambaSL')
    parser.add_argument('--use_D', type=int, default=0, help='whether to use D for MambaSL')
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
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'radam'])
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_epochs', type=int, default=1)
    parser.add_argument('--weight_averaging', type=str, default='none', choices=['none', 'ema', 'swa', 'ema_swa'])
    parser.add_argument('--ema_decay', type=float, default=0.995)
    parser.add_argument('--ema_start_epoch', type=int, default=1)
    parser.add_argument('--swa_start_epoch', type=int, default=16)
    parser.add_argument('--swa_end_epoch', type=int, default=0)
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--loss_schedule', type=str, default='', help='epoch loss schedule, e.g. SMAPE:20,MASE:30')
    parser.add_argument('--huber_delta', type=float, default=1.0, help='SmoothL1/Huber loss beta')
    parser.add_argument('--loss_mse_weight', type=float, default=0.5, help='MSE weight for MSEMAE loss')
    parser.add_argument('--loss_variable_weights', type=str, default='')
    parser.add_argument('--loss_horizon_weight', type=float, default=1.0)
    parser.add_argument('--loss_horizon_weight_start', type=int, default=0)
    parser.add_argument('--loss_horizon_weight_mode', type=str, default='step')
    parser.add_argument('--loss_volatility_weight', type=float, default=0.0)
    parser.add_argument('--loss_level_weight', type=float, default=0.0)
    parser.add_argument('--loss_variance_weight', type=float, default=0.0)
    parser.add_argument('--loss_range_weight', type=float, default=0.0)
    parser.add_argument('--loss_tail_bias_weight', type=float, default=0.0)
    parser.add_argument('--loss_tail_bias_start', type=int, default=0)
    parser.add_argument('--loss_tail_bias_variables', type=str, default='')
    parser.add_argument('--loss_tail_level_weight', type=float, default=0.0)
    parser.add_argument('--loss_tail_level_start', type=int, default=0)
    parser.add_argument('--loss_tail_level_variables', type=str, default='')
    parser.add_argument('--loss_tail_hard_weight', type=float, default=0.0)
    parser.add_argument('--loss_tail_hard_start', type=int, default=0)
    parser.add_argument('--loss_tail_hard_power', type=float, default=1.0)
    parser.add_argument('--loss_tail_hard_clip', type=float, default=3.0)
    parser.add_argument('--loss_tail_lowpass_weight', type=float, default=0.0)
    parser.add_argument('--loss_tail_lowpass_start', type=int, default=0)
    parser.add_argument('--loss_tail_lowpass_kernel', type=int, default=9)
    parser.add_argument('--loss_tail_lowpass_variables', type=str, default='')
    parser.add_argument('--loss_consistency_weight', type=float, default=0.0)
    parser.add_argument('--loss_consistency_variables', type=str, default='')
    parser.add_argument('--vali_metric_mode', type=str, default='all')
    parser.add_argument('--vali_metric_horizon_start', type=int, default=0)
    parser.add_argument('--validation_only', action='store_true', default=False)
    parser.add_argument('--strict_checkpoint', action='store_true', default=False)
    parser.add_argument('--save_epoch_checkpoints', action='store_true', default=False)
    parser.add_argument('--save_epoch_start', type=int, default=1)
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--max_train_steps', type=int, default=0,
                        help='stop after this many optimizer steps; 0 disables step budgeting')
    parser.add_argument('--stop_after_epochs', type=int, default=0,
                        help='stop after completing this many train/vali/test epochs; 0 disables')
    parser.add_argument('--test_every_epoch', type=int, default=1,
                        help='evaluate full test metrics after every epoch and save epoch_test_metrics.csv')
    parser.add_argument('--select_best_by_test_metric', type=str, default='',
                        help='optionally save checkpoint by per-epoch test metric: mse, mae, rmse, mape, or mspe')
    parser.add_argument('--classification_eval_steps', type=int, default=0,
                        help='evaluate classification TEST split every N optimizer steps; 0 disables step-level eval')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

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
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='enable deterministic CUDA algorithms for reproducibility')
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

    # CANPatchTST
    parser.add_argument('--can_stride', type=int, default=8, help='CAN patch stride')
    parser.add_argument('--can_shifts', type=str, default='1,2,4,8', help='CAN channel shift list')
    parser.add_argument('--can_temporal_shifts', type=str, default='', help='CAN temporal shift list')
    parser.add_argument('--can_cli_mode', type=str, default='full', help='CAN channel interaction mode')
    parser.add_argument('--can_temporal_cli_mode', type=str, default='inner', help='CAN temporal interaction mode')
    parser.add_argument('--can_ctx_mode', type=str, default='diff', help='CAN context mode')
    parser.add_argument('--can_global_cli_mode', type=str, default='inner', help='CAN global interaction mode')
    parser.add_argument('--can_global_ctx_mode', type=str, default='abs', help='CAN global context mode')
    parser.add_argument('--can_global_shifts', type=str, default='', help='CAN global shift list')
    parser.add_argument('--can_kernel_size', type=int, default=3, help='CAN depthwise kernel size')
    parser.add_argument('--can_drop_path', type=float, default=0.1, help='CAN stochastic depth rate')
    parser.add_argument('--can_drop_path_schedule', type=str, default='linear', help='CAN stochastic depth schedule')
    parser.add_argument('--can_init_values', type=float, default=1e-5, help='CAN layer scale init')
    parser.add_argument('--can_gamma_lr_scale', type=float, default=1.0, help='CAN gamma learning-rate scale')
    parser.add_argument('--can_gamma_weight_decay', type=float, default=0.0, help='CAN gamma weight decay')
    parser.add_argument('--can_beta_init', type=float, default=0.5, help='CAN beta init')
    parser.add_argument('--can_temporal_beta_init', type=float, default=None, help='CAN temporal beta init')
    parser.add_argument('--can_global_beta_init', type=float, default=None, help='CAN global beta init')
    parser.add_argument('--can_temporal_roll', type=int, default=1, help='enable CAN temporal interaction')
    parser.add_argument('--can_temporal_circular', type=int, default=0, help='use circular temporal shifts')
    parser.add_argument('--can_context_pyramid', type=int, default=0, help='enable CAN context pyramid')
    parser.add_argument('--can_use_gffng', type=int, default=1, help='enable CAN global FFN gate')
    parser.add_argument('--can_use_ffn', type=int, default=0, help='enable CAN local FFN')
    parser.add_argument('--can_use_orth', type=int, default=0, help='enable CAN orthogonal context')
    parser.add_argument('--can_cross_var', type=int, default=0, help='enable CAN cross-variable blocks')
    parser.add_argument('--can_cross_var_layers', type=int, default=1, help='CAN cross-variable layers')
    parser.add_argument('--can_cross_var_context', type=str, default='others_mean', help='CAN cross-variable context')
    parser.add_argument('--can_cross_var_shifts', type=str, default='1,2,4,8,16', help='CAN cross-variable shifts')
    parser.add_argument('--can_var_attn', type=int, default=0, help='enable CAN variable attention')
    parser.add_argument('--can_var_attn_layers', type=int, default=1, help='CAN variable attention layers')
    parser.add_argument('--can_var_attn_dim', type=int, default=32, help='CAN variable attention dim')
    parser.add_argument('--can_var_attn_top_k', type=int, default=0, help='CAN variable attention top-k')
    parser.add_argument('--can_var_attn_shifts', type=str, default='1,2,4,8', help='CAN variable attention shifts')
    parser.add_argument('--can_var_embed', type=int, default=0, help='enable CAN variable embedding')
    parser.add_argument('--can_time_mark', type=int, default=0, help='enable CAN time-mark branch')
    parser.add_argument('--can_time_mark_mode', type=str, default='flatten', help='CAN time-mark mode')
    parser.add_argument('--can_time_mark_scale_init', type=float, default=1.0, help='CAN time-mark scale init')
    parser.add_argument('--can_linear_residual', type=int, default=0, help='enable CAN linear residual')
    parser.add_argument('--can_linear_mode', type=str, default='raw', help='CAN linear residual mode')
    parser.add_argument('--can_linear_individual', type=int, default=0, help='use per-variable linear residual')
    parser.add_argument('--can_linear_scale_init', type=float, default=0.5, help='CAN linear residual scale init')
    parser.add_argument('--can_periodic_residual', type=int, default=0, help='enable CAN periodic residual')
    parser.add_argument('--can_periods', type=str, default='24', help='CAN residual periods')
    parser.add_argument('--can_periodic_alpha', type=float, default=0.2, help='CAN periodic residual scale')
    parser.add_argument('--can_periodic_learnable', type=int, default=0, help='learn CAN periodic residual scale')
    parser.add_argument('--can_coarse_var_attn', type=int, default=0, help='enable CAN coarse variable attention')
    parser.add_argument('--can_coarse_var_levels', type=int, default=3, help='CAN coarse variable levels')
    parser.add_argument('--can_coarse_var_dim', type=int, default=32, help='CAN coarse variable dim')
    parser.add_argument('--can_coarse_var_scale_init', type=float, default=0.1, help='CAN coarse variable scale init')
    parser.add_argument('--can_coarse_var_mode', type=str, default='diff', help='CAN coarse variable mode')
    parser.add_argument('--can_hierarchical_mixer', type=int, default=0, help='enable CAN hierarchical mixer')
    parser.add_argument('--can_hierarchical_levels', type=int, default=3, help='CAN hierarchical levels')
    parser.add_argument('--can_hierarchical_layers', type=int, default=1, help='CAN hierarchical layers')
    parser.add_argument('--can_hierarchical_dim', type=int, default=64, help='CAN hierarchical dim')
    parser.add_argument('--can_hierarchical_cross_scale_init', type=float, default=0.05, help='CAN hierarchical cross scale')
    parser.add_argument('--can_hierarchical_fusion_init', type=float, default=0.2, help='CAN hierarchical fusion scale')
    parser.add_argument('--can_hierarchical_mode', type=str, default='blend', help='CAN hierarchical mode')
    parser.add_argument('--can_hierarchical_residual_scale_init', type=float, default=1.0, help='CAN hierarchical residual scale')
    parser.add_argument('--can_periodic_image', type=int, default=0, help='enable CAN periodic image branch')
    parser.add_argument('--can_periodic_image_top_k', type=int, default=3, help='CAN periodic image top-k')
    parser.add_argument('--can_periodic_image_dim', type=int, default=32, help='CAN periodic image dim')
    parser.add_argument('--can_periodic_image_layers', type=int, default=1, help='CAN periodic image layers')
    parser.add_argument('--can_periodic_image_shifts', type=str, default='1,2,4', help='CAN periodic image shifts')
    parser.add_argument('--can_periodic_image_scale_init', type=float, default=0.0, help='CAN periodic image scale init')
    parser.add_argument('--can_deep_periodic_image', type=int, default=0, help='enable CAN deep periodic image branch')
    parser.add_argument('--can_deep_periodic_top_k', type=int, default=3, help='CAN deep periodic top-k')
    parser.add_argument('--can_deep_periodic_layers', type=int, default=1, help='CAN deep periodic layers')
    parser.add_argument('--can_deep_periodic_shifts', type=str, default='1,2,4', help='CAN deep periodic shifts')
    parser.add_argument('--can_deep_periodic_scale_init', type=float, default=0.1, help='CAN deep periodic scale init')
    parser.add_argument('--can_multiscale_patch_lens', type=str, default='', help='CAN multiscale patch lengths')
    parser.add_argument('--can_multiscale_stride_ratio', type=float, default=0.5, help='CAN multiscale stride ratio')
    parser.add_argument('--can_multiscale_main_bias', type=float, default=0.0, help='CAN multiscale main branch bias')
    parser.add_argument('--can_cls_head', type=str, default='flatten', choices=['flatten', 'pool', 'var_pool'],
                        help='CAN classification head')

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
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
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
            
            # Override setting for specific model to ensure proper checkpoint naming and logging
            if args.model == 'MambaSingleLayer' and args.task_name == 'classification':
                setting = f'{args.task_name}_CLS_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                        + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_ds{args.d_ff}' \
                        + f'_expand{args.expand}_dc{args.d_conv}_nk{args.num_kernels}' \
                        + f'_tvdt{int(args.tv_dt)}_tvB{int(args.tv_B)}_tvC{int(args.tv_C)}_useD{int(args.use_D)}_{args.des}_{ii}'

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
        
        # Override setting for specific model to ensure proper checkpoint naming and logging
        if args.model == 'MambaSingleLayer' and args.task_name == 'classification':
            setting = f'{args.task_name}_CLS_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                    + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_ds{args.d_ff}' \
                    + f'_expand{args.expand}_dc{args.d_conv}_nk{args.num_kernels}' \
                    + f'_tvdt{args.tv_dt}_tvB{args.tv_B}_tvC{args.tv_C}_useD{int(args.use_D)}_{args.des}_{ii}'

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.use_gpu:
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
