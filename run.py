import argparse
import random

import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args


def build_parser():
    parser = argparse.ArgumentParser(description="TS-CAN for long-term time series forecasting")

    parser.add_argument("--task_name", type=str, default="long_term_forecast", choices=["long_term_forecast"])
    parser.add_argument("--is_training", type=int, default=1)
    parser.add_argument("--model_id", type=str, default="TSCAN_ETTh1_96_best")
    parser.add_argument("--model", type=str, default="CANPatchTST", choices=["CANPatchTST"])

    parser.add_argument(
        "--data",
        type=str,
        default="ETTh1",
        choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "custom"],
    )
    parser.add_argument("--root_path", type=str, default="./dataset/ETT/")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--features", type=str, default="M", choices=["M", "S", "MS"])
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--results", type=str, default="./results/")
    parser.add_argument("--test_results", type=str, default="./test_results/")

    parser.add_argument("--seq_len", type=int, default=192)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--inverse", action="store_true", default=False)

    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--dec_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--distil", action="store_false", default=True)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_kernels", type=int, default=6)
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly")

    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--can_stride", type=int, default=8)
    parser.add_argument("--can_shifts", type=str, default="1,2,4,8,16")
    parser.add_argument(
        "--can_cli_mode",
        type=str,
        default="full",
        choices=["full", "inner", "wedge", "adaptive"],
    )
    parser.add_argument(
        "--can_temporal_cli_mode",
        type=str,
        default="full",
        choices=["full", "inner", "wedge", "adaptive"],
    )
    parser.add_argument("--can_ctx_mode", type=str, default="diff", choices=["diff", "abs"])
    parser.add_argument("--can_drop_path", type=float, default=0.05)
    parser.add_argument("--can_kernel_size", type=int, default=3)
    parser.add_argument("--can_init_values", type=float, default=1e-5)
    parser.add_argument("--can_use_gffng", type=int, default=1)
    parser.add_argument("--can_temporal_roll", type=int, default=1)
    parser.add_argument("--can_beta_init", type=float, default=0.5)
    parser.add_argument("--can_use_orth", type=int, default=0)
    parser.add_argument("--can_context_pyramid", type=int, default=0)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--des", type=str, default="TSCAN_RELEASE")
    parser.add_argument("--loss", type=str, default="MSE")
    parser.add_argument("--lradj", type=str, default="cosine", choices=["type1", "type2", "type3", "cosine"])
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--use_dtw", action="store_true", default=False)

    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--no_use_gpu", action="store_false", dest="use_gpu")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu_type", type=str, default="cuda", choices=["cuda", "mps"])
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0")

    parser.add_argument("--p_hidden_dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--p_hidden_layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2)

    return parser


def build_setting(args, run_idx):
    return (
        f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_"
        f"ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_"
        f"dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_"
        f"df{args.d_ff}_expand{args.expand}_dc{args.d_conv}_fc{args.factor}_"
        f"eb{args.embed}_dt{args.distil}_{args.des}_{run_idx}"
    )


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    set_random_seed(args.seed)

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device(f"cuda:{args.gpu}")
        print("Using GPU")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and args.use_gpu:
        args.device = torch.device("mps")
        print("Using MPS")
    else:
        args.device = torch.device("cpu")
        print("Using CPU")

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        args.device_ids = [int(device_id) for device_id in args.devices.split(",")]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print_args(args)

    Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            setting = build_setting(args, ii)
            print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            exp.train(setting)
            print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.test(setting)
            if args.use_gpu and args.device.type == "cuda":
                torch.cuda.empty_cache()
    else:
        exp = Exp(args)
        setting = build_setting(args, 0)
        print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(setting, test=1)
        if args.use_gpu and args.device.type == "cuda":
            torch.cuda.empty_cache()
