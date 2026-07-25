#!/usr/bin/env python
"""
TS-CAN standalone entry for Time-Series-Library-main.
Reuses parent TS-CAN-github infrastructure. TSLib files untouched.
"""
import os, sys, argparse, random
import numpy as np, torch, importlib

_PARENT = os.path.dirname(os.path.abspath(__file__))  # Time-Series-Library-main
_CAN   = os.path.dirname(_PARENT)                      # TS-CAN-github
sys.path.insert(0, _CAN)

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args

# Parent infrastructure is imported above. Model modules and their layer
# dependencies must resolve from TSLib, which contains the full model zoo.
# The shared experiment runner imports the parent ``utils`` package first.
# Extend that package path so TSLib-only modules such as ``utils.masking``
# remain importable without replacing the shared parent utilities.
import utils as _shared_utils
_TSLIB_UTILS = os.path.join(_PARENT, "utils")
if hasattr(_shared_utils, "__path__") and _TSLIB_UTILS not in _shared_utils.__path__:
    _shared_utils.__path__.append(_TSLIB_UTILS)

if _PARENT in sys.path:
    sys.path.remove(_PARENT)
sys.path.insert(0, _PARENT)

# --- model auto-discovery from Time-Series-Library-main/models/ ---
_MODEL_MAP = {}
_md = os.path.join(_PARENT, "models")
if os.path.isdir(_md):
    for fn in os.listdir(_md):
        if fn.endswith(".py") and fn != "__init__.py":
            _MODEL_MAP[fn[:-3]] = f"models.{fn[:-3]}"

class _Lazy(dict):
    def __getitem__(self, k):
        if k in self: return super().__getitem__(k)
        mod = importlib.import_module(_MODEL_MAP[k])
        cls = mod.Model if hasattr(mod, "Model") else getattr(mod, k)
        self[k] = cls
        return cls

Exp_Long_Term_Forecast.model_dict = _Lazy()


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--task_name", type=str, required=True, default="long_term_forecast")
    p.add_argument("--is_training", type=int, required=True, default=1)
    p.add_argument("--model_id", type=str, required=True, default="CAN_ETTh1_96")
    p.add_argument("--model", type=str, required=True, default="CANPatchTST")
    p.add_argument("--data", type=str, required=True, default="ETTh1")
    p.add_argument("--root_path", type=str, default="./dataset/ETT-small/")
    p.add_argument("--data_path", type=str, default="ETTh1.csv")
    p.add_argument("--features", type=str, default="M")
    p.add_argument("--target", type=str, default="OT")
    p.add_argument("--freq", type=str, default="h")
    p.add_argument("--checkpoints", type=str, default="./checkpoints/")
    p.add_argument("--results", type=str, default="./results/")
    p.add_argument("--test_results", type=str, default="./test_results/")
    p.add_argument("--test_checkpoint", type=str, default="")
    p.add_argument("--strict_checkpoint", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=192)
    p.add_argument("--label_len", type=int, default=48)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--inverse", action="store_true", default=False)
    p.add_argument("--enc_in", type=int, default=7)
    p.add_argument("--dec_in", type=int, default=7)
    p.add_argument("--c_out", type=int, default=7)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--e_layers", type=int, default=2)
    p.add_argument("--d_layers", type=int, default=1)
    p.add_argument("--d_ff", type=int, default=192)
    p.add_argument("--moving_avg", type=int, default=25)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--embed", type=str, default="timeF")
    p.add_argument("--activation", type=str, default="gelu")
    p.add_argument("--factor", type=int, default=1)
    p.add_argument("--distil", action="store_false", default=True)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--d_conv", type=int, default=4)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--num_kernels", type=int, default=6)
    p.add_argument("--seasonal_patterns", type=str, default="Monthly")
    p.add_argument("--channel_independence", type=int, default=1)
    p.add_argument(
        "--decomp_method",
        type=str,
        default="moving_avg",
        choices=["moving_avg", "dft_decomp"]
    )
    p.add_argument("--use_norm", type=int, default=1)
    p.add_argument("--down_sampling_layers", type=int, default=0)
    p.add_argument("--down_sampling_window", type=int, default=1)
    p.add_argument(
        "--down_sampling_method",
        type=str,
        default=None,
        choices=["avg", "max", "conv"]
    )
    # CAN
    p.add_argument("--patch_len", type=int, default=16)
    p.add_argument("--can_stride", type=int, default=8)
    p.add_argument("--can_shifts", type=str, default="1,2,4,8,16")
    p.add_argument("--can_temporal_shifts", type=str, default="")
    p.add_argument("--can_cli_mode", type=str, default="full")
    p.add_argument("--can_temporal_cli_mode", type=str, default="full")
    p.add_argument("--can_ctx_mode", type=str, default="diff")
    p.add_argument("--can_drop_path", type=float, default=0.05)
    p.add_argument(
        "--can_drop_path_schedule",
        type=str,
        default="linear",
        choices=["linear", "uniform"],
    )
    p.add_argument("--can_gamma_lr_scale", type=float, default=1.0)
    p.add_argument("--can_gamma_weight_decay", type=float, default=0.0)
    p.add_argument("--can_kernel_size", type=int, default=3)
    p.add_argument("--can_init_values", type=float, default=1e-5)
    p.add_argument("--can_use_gffng", type=int, default=1)
    p.add_argument("--can_global_cli_mode", type=str, default="inner")
    p.add_argument("--can_global_ctx_mode", type=str, default="abs")
    p.add_argument("--can_global_shifts", type=str, default="")
    p.add_argument("--can_temporal_roll", type=int, default=1)
    p.add_argument("--can_temporal_circular", type=int, default=0)
    p.add_argument("--can_beta_init", type=float, default=0.5)
    p.add_argument("--can_temporal_beta_init", type=float, default=None)
    p.add_argument("--can_global_beta_init", type=float, default=None)
    p.add_argument("--can_use_orth", type=int, default=0)
    p.add_argument("--can_context_pyramid", type=int, default=1)
    p.add_argument("--can_use_ffn", type=int, default=0)
    p.add_argument("--can_cross_var", type=int, default=0)
    p.add_argument("--can_cross_var_layers", type=int, default=1)
    p.add_argument("--can_cross_var_context", type=str, default="others_mean")
    p.add_argument("--can_cross_var_shifts", type=str, default="1,2,4,8,16")
    p.add_argument("--can_var_embed", type=int, default=0)
    p.add_argument("--can_time_mark", type=int, default=0)
    p.add_argument("--can_time_mark_mode", type=str, default="flatten")
    p.add_argument("--can_time_mark_scale_init", type=float, default=1.0)
    p.add_argument("--can_linear_residual", type=int, default=0)
    p.add_argument("--can_linear_mode", type=str, default="raw")
    p.add_argument("--can_linear_individual", type=int, default=0)
    p.add_argument("--can_linear_scale_init", type=float, default=0.5)
    p.add_argument("--can_periodic_residual", type=int, default=0)
    p.add_argument("--can_periods", type=str, default="24")
    p.add_argument("--can_periodic_alpha", type=float, default=0.2)
    p.add_argument("--can_periodic_learnable", type=int, default=0)
    p.add_argument("--can_coarse_var_attn", type=int, default=0)
    p.add_argument("--can_coarse_var_levels", type=int, default=3)
    p.add_argument("--can_coarse_var_dim", type=int, default=32)
    p.add_argument("--can_coarse_var_scale_init", type=float, default=0.1)
    p.add_argument("--can_coarse_var_mode", type=str, default="diff", choices=["abs", "diff"])
    p.add_argument("--can_hierarchical_mixer", type=int, default=0)
    p.add_argument("--can_hierarchical_levels", type=int, default=3)
    p.add_argument("--can_hierarchical_layers", type=int, default=1)
    p.add_argument("--can_hierarchical_dim", type=int, default=64)
    p.add_argument("--can_hierarchical_cross_scale_init", type=float, default=0.05)
    p.add_argument("--can_hierarchical_fusion_init", type=float, default=0.2)
    p.add_argument(
        "--can_hierarchical_mode",
        type=str,
        default="blend",
        choices=["blend", "residual"]
    )
    p.add_argument("--can_hierarchical_residual_scale_init", type=float, default=1.0)
    p.add_argument("--can_periodic_image", type=int, default=0)
    p.add_argument("--can_periodic_image_top_k", type=int, default=3)
    p.add_argument("--can_periodic_image_dim", type=int, default=32)
    p.add_argument("--can_periodic_image_layers", type=int, default=1)
    p.add_argument("--can_periodic_image_shifts", type=str, default="1,2,4")
    p.add_argument("--can_periodic_image_scale_init", type=float, default=0.0)
    p.add_argument("--can_deep_periodic_image", type=int, default=0)
    p.add_argument("--can_deep_periodic_top_k", type=int, default=3)
    p.add_argument("--can_deep_periodic_layers", type=int, default=1)
    p.add_argument("--can_deep_periodic_shifts", type=str, default="1,2,4")
    p.add_argument("--can_deep_periodic_scale_init", type=float, default=0.1)
    p.add_argument("--can_multiscale_patch_lens", type=str, default="")
    p.add_argument("--can_multiscale_stride_ratio", type=float, default=0.5)
    p.add_argument("--can_multiscale_main_bias", type=float, default=0.0)
    # CAN v2
    p.add_argument("--can_v2_group_size", type=int, default=4)
    p.add_argument("--can_v2_variable_top_k", type=int, default=0)
    p.add_argument("--can_v2_variable_init", type=float, default=1e-4)
    p.add_argument("--can_v2_temporal_top_k", type=int, default=0)
    p.add_argument("--can_v2_scale_factors", type=str, default="1,2,4")
    p.add_argument("--can_v2_main_scale_bias", type=float, default=4.0)
    # CAN v3
    p.add_argument("--can_v3_scale_factors", type=str, default="1,2,4")
    p.add_argument("--can_v3_variable_context", type=int, default=1)
    p.add_argument("--can_v3_cross_scale", type=int, default=1)
    p.add_argument("--can_v3_cross_scale_init", type=float, default=1e-4)
    p.add_argument("--can_v3_decomp_kernel", type=int, default=3)
    p.add_argument("--can_v3_main_scale_bias", type=float, default=2.0)
    # Periodic time-image CAN
    p.add_argument("--can_image_top_k", type=int, default=3)
    p.add_argument("--can_image_min_period", type=int, default=4)
    p.add_argument("--can_image_mixerpp_scales", type=int, default=4)
    p.add_argument(
        "--can_image_period_source",
        type=str,
        default="local",
        choices=["local", "coarse_shared"],
    )
    p.add_argument(
        "--can_image_output_mode",
        type=str,
        default="absolute",
        choices=["absolute", "last_residual", "last_trend"],
    )
    p.add_argument("--can_image_output_trend_tail", type=int, default=24)
    p.add_argument("--can_image_encode_scales", type=str, default="")
    p.add_argument("--can_image_encode_scale_init", type=float, default=0.0)
    p.add_argument("--can_image_multiscale_predict", type=int, default=0)
    p.add_argument("--can_image_predict_scales", type=str, default="2,4")
    p.add_argument("--can_image_multiscale_predict_init", type=float, default=0.1)
    p.add_argument("--can_image_decomp_scale_mixer", type=int, default=0)
    p.add_argument("--can_image_decomp_scale_mixer_scales", type=str, default="1,2,4")
    p.add_argument("--can_image_decomp_scale_mixer_init", type=float, default=0.03)
    p.add_argument("--can_image_pdm_backbone", type=int, default=0)
    p.add_argument("--can_image_pdm_backbone_scales", type=str, default="1,2,4,8")
    p.add_argument("--can_image_pdm_backbone_init", type=float, default=1.0)
    p.add_argument("--can_image_pdm_backbone_layers", type=int, default=1)
    p.add_argument("--can_image_pdm_scale_predict", type=int, default=0)
    p.add_argument("--can_image_pdm_scale_predict_init", type=float, default=0.1)
    p.add_argument(
        "--can_image_pdm_backbone_mode",
        type=str,
        default="replace",
        choices=["replace", "residual"],
    )
    p.add_argument("--can_image_dual_axis_decomp", type=int, default=0)
    p.add_argument("--can_image_dual_axis_init", type=float, default=1e-4)
    p.add_argument("--can_image_dual_axis_scale", type=float, default=1.0)
    p.add_argument("--can_image_dual_axis_scale_predict", type=int, default=0)
    p.add_argument("--can_image_dual_axis_scale_predict_init", type=float, default=0.05)
    p.add_argument(
        "--can_image_dual_axis_mode",
        type=str,
        default="pdm",
        choices=["pdm", "residual", "replace"],
    )
    p.add_argument("--can_image_horizon_extend", type=int, default=0)
    p.add_argument("--can_image_horizon_extend_init", type=float, default=0.05)
    p.add_argument("--can_image_decomp_encode", type=int, default=0)
    p.add_argument("--can_image_decomp_kernel", type=int, default=25)
    p.add_argument(
        "--can_image_decomp_method",
        type=str,
        default="moving_avg",
        choices=["moving_avg", "dft"],
    )
    p.add_argument("--can_image_decomp_top_k", type=int, default=5)
    p.add_argument("--can_image_decomp_scale_init", type=float, default=1e-3)
    p.add_argument("--can_image_decomp_predict", type=int, default=0)
    p.add_argument(
        "--can_image_decomp_predict_mode",
        type=str,
        default="season_trend",
        choices=["season_trend", "trend_residual"],
    )
    p.add_argument("--can_image_decomp_trend_kernels", type=str, default="")
    p.add_argument("--can_image_decomp_trend_scale_init", type=float, default=1.0)
    p.add_argument("--can_image_decomp_season_scale_init", type=float, default=1.0)
    p.add_argument(
        "--can_image_decomp_fusion",
        type=str,
        default="add",
        choices=["add", "gated", "residual_gate"],
    )
    p.add_argument("--can_image_decomp_fusion_init", type=float, default=0.35)
    p.add_argument("--can_image_head", type=str, default="flatten")
    p.add_argument("--can_image_head_layers", type=int, default=1)
    p.add_argument("--can_image_prompt_layers", type=int, default=1)
    p.add_argument("--can_image_prompt_scales", type=str, default="1,2,4")
    p.add_argument("--can_image_prompt_use_local", type=int, default=1)
    p.add_argument("--can_image_segment_len", type=int, default=24)
    p.add_argument("--can_image_semi_ar_hidden_scale", type=float, default=1.0)
    p.add_argument("--can_image_horizon_raw_init", type=float, default=0.0)
    p.add_argument("--can_image_horizon_prompt_init", type=float, default=1.0)
    p.add_argument("--can_image_residual_init", type=float, default=1e-3)
    p.add_argument("--can_image_residual_scales", type=str, default="2,4,8")
    p.add_argument("--can_image_residual_start", type=int, default=48)
    p.add_argument("--can_image_residual_use_prompt", type=int, default=1)
    p.add_argument("--can_image_residual_base", type=str, default="")
    p.add_argument("--can_image_tail_residual_start", type=int, default=48)
    p.add_argument("--can_image_tail_residual_ramp", type=int, default=24)
    p.add_argument("--can_image_innovation_init", type=float, default=0.3)
    p.add_argument("--can_image_innovation_tail", type=int, default=48)
    p.add_argument("--can_image_innovation_kernel", type=int, default=13)
    p.add_argument("--can_image_innovation_use_raw", type=int, default=0)
    p.add_argument("--can_image_trend_init", type=float, default=0.02)
    p.add_argument("--can_image_trend_tail", type=int, default=24)
    p.add_argument("--can_image_trend_start", type=int, default=64)
    p.add_argument("--can_image_forecast_gate_mode", type=str, default="full")
    p.add_argument("--can_image_forecast_gate_init", type=float, default=1.0)
    p.add_argument("--can_image_stats_tail", type=int, default=24)
    p.add_argument("--can_image_stats_init", type=float, default=0.1)
    p.add_argument("--can_image_stats_hidden", type=int, default=128)
    p.add_argument(
        "--can_image_stats_mode",
        type=str,
        default="gate",
        choices=["gate", "residual", "both", "context", "prompt", "full"]
    )
    p.add_argument("--can_time_covariates", type=int, default=0)
    p.add_argument("--can_time_covariate_dim", type=int, default=4)
    p.add_argument("--can_image_internal_aux", type=int, default=0)
    p.add_argument("--can_image_internal_aux_kernel", type=int, default=9)
    p.add_argument("--can_image_internal_aux_init", type=float, default=1.0)
    p.add_argument("--can_image_cross_var", type=int, default=0)
    p.add_argument("--can_image_cross_var_shifts", type=str, default="1,2,3")
    p.add_argument("--can_image_cross_var_mode", type=str, default="full")
    p.add_argument("--can_image_cross_var_ctx_mode", type=str, default="diff")
    p.add_argument("--can_image_cross_var_init", type=float, default=1e-3)
    p.add_argument("--can_image_cross_var_mixer", type=int, default=0)
    p.add_argument("--can_image_cross_var_mixer_shifts", type=str, default="1,2,3")
    p.add_argument("--can_image_cross_var_mixer_mode", type=str, default="full")
    p.add_argument("--can_image_cross_var_mixer_ctx_mode", type=str, default="diff")
    p.add_argument("--can_image_cross_var_mixer_init", type=float, default=1e-3)
    p.add_argument("--can_image_cross_var_mixer_layers", type=int, default=1)
    p.add_argument("--can_image_coarse_gate_init", type=float, default=0.0)
    p.add_argument("--can_v215_decomp_kernel", type=int, default=13)
    p.add_argument("--can_v215_detail_scale_init", type=float, default=0.05)
    p.add_argument("--can_v215_update_scale_init", type=float, default=0.5)
    p.add_argument("--can_image_cross_var_graph", type=int, default=0)
    p.add_argument("--can_image_cross_var_graph_top_k", type=int, default=3)
    p.add_argument("--can_image_cross_var_graph_mode", type=str, default="full")
    p.add_argument("--can_image_cross_var_graph_ctx_mode", type=str, default="diff")
    p.add_argument("--can_image_cross_var_graph_init", type=float, default=1e-4)
    p.add_argument("--can_image_cross_var_graph_layers", type=int, default=1)
    p.add_argument("--can_image_cross_var_graph_temperature", type=float, default=1.0)
    p.add_argument("--can_image_lowpass_residual", type=int, default=0)
    p.add_argument("--can_image_lowpass_scales", type=str, default="2,4,8")
    p.add_argument("--can_image_lowpass_init", type=float, default=1e-3)
    p.add_argument("--can_image_lowpass_start", type=int, default=64)
    p.add_argument("--can_image_linear_decomp_residual", type=int, default=0)
    p.add_argument("--can_image_linear_decomp_init", type=float, default=1e-3)
    p.add_argument("--can_image_linear_decomp_kernel", type=int, default=25)
    p.add_argument("--can_image_linear_decomp_start", type=int, default=0)
    p.add_argument("--can_image_linear_decomp_gate", type=int, default=1)
    p.add_argument("--can_image_level_anchor", type=int, default=0)
    p.add_argument("--can_image_level_init", type=float, default=1e-3)
    p.add_argument("--can_image_level_tail", type=int, default=48)
    p.add_argument("--can_image_level_hidden", type=int, default=32)
    p.add_argument("--can_image_level_start", type=int, default=0)
    p.add_argument("--can_image_tail_trend_stabilizer", type=int, default=0)
    p.add_argument("--can_image_tail_trend_init", type=float, default=1e-3)
    p.add_argument("--can_image_tail_trend_scales", type=str, default="2,4,8")
    p.add_argument("--can_image_tail_trend_tail", type=int, default=48)
    p.add_argument("--can_image_tail_trend_hidden", type=int, default=32)
    p.add_argument("--can_image_tail_trend_start", type=int, default=96)
    p.add_argument("--can_image_level_memory_can", type=int, default=0)
    p.add_argument("--can_image_level_memory_init", type=float, default=1e-3)
    p.add_argument("--can_image_level_memory_tail", type=int, default=48)
    p.add_argument("--can_image_level_memory_hidden", type=int, default=32)
    p.add_argument("--can_image_level_memory_start", type=int, default=96)
    p.add_argument("--can_image_level_memory_shifts", type=str, default="1,2,3")
    p.add_argument("--can_image_level_memory_mode", type=str, default="full")
    p.add_argument("--can_image_level_memory_ctx_mode", type=str, default="diff")
    p.add_argument("--can_image_amp_calibrator", type=int, default=0)
    p.add_argument("--can_image_amp_init", type=float, default=0.0)
    p.add_argument("--can_image_amp_tail", type=int, default=48)
    p.add_argument("--can_image_amp_hidden", type=int, default=32)
    p.add_argument("--can_image_amp_start", type=int, default=48)
    p.add_argument("--can_long_aux", type=int, default=0)
    p.add_argument("--can_long_aux_init", type=float, default=0.01)
    p.add_argument("--can_long_aux_hidden", type=int, default=32)
    p.add_argument("--can_long_aux_kernel", type=int, default=25)
    p.add_argument("--can_long_aux_trend_scales", type=str, default="1")
    p.add_argument("--can_long_aux_memory_scales", type=str, default="1,2,4,8")
    p.add_argument("--can_long_aux_memory_tokens", type=int, default=8)
    p.add_argument("--can_long_aux_memory_shifts", type=str, default="1,2,3")
    p.add_argument("--can_long_aux_memory_init", type=float, default=0.01)
    p.add_argument("--can_long_aux_detail_scales", type=str, default="1,2,4,8")
    p.add_argument("--can_long_aux_detail_tokens", type=int, default=8)
    p.add_argument("--can_long_aux_detail_shifts", type=str, default="1,2,3")
    p.add_argument("--can_long_aux_detail_init", type=float, default=0.001)
    p.add_argument("--can_long_aux_detail_start", type=int, default=1)
    p.add_argument("--can_long_aux_detail_ramp", type=int, default=1)
    p.add_argument("--can_long_aux_detail_learned_gate", type=int, default=0)
    p.add_argument("--can_long_aux_detail_learned_gate_init", type=float, default=2.0)
    p.add_argument("--can_long_aux_amp_residual", type=int, default=0)
    p.add_argument("--can_long_aux_amp_init", type=float, default=0.0)
    p.add_argument("--can_long_aux_amp_start", type=int, default=169)
    p.add_argument("--can_long_aux_amp_ramp", type=int, default=48)
    p.add_argument("--can_long_aux_var_residual", type=int, default=0)
    p.add_argument("--can_long_aux_var_init", type=float, default=0.0)
    p.add_argument("--can_long_aux_var_scales", type=str, default="1,2,4,8")
    p.add_argument("--can_long_aux_var_start", type=int, default=169)
    p.add_argument("--can_long_aux_var_ramp", type=int, default=48)
    p.add_argument("--can_long_aux_pred_calibrator", type=int, default=0)
    p.add_argument("--can_long_aux_pred_amp_init", type=float, default=0.0)
    p.add_argument("--can_long_aux_pred_bias_init", type=float, default=0.0)
    p.add_argument("--can_long_aux_pred_calib_start", type=int, default=97)
    p.add_argument("--can_long_aux_pred_calib_ramp", type=int, default=48)
    p.add_argument("--can_long_state_inject", type=int, default=0)
    p.add_argument(
        "--can_long_state_mode",
        type=str,
        default="trend_detail",
        choices=["trend", "detail", "trend_detail"],
    )
    p.add_argument("--can_long_state_init", type=float, default=1e-4)
    p.add_argument("--can_long_state_kernel", type=int, default=25)
    p.add_argument("--can_long_state_shifts", type=str, default="1,2,3")
    p.add_argument("--can_long_state_cli_mode", type=str, default="full")
    p.add_argument("--can_long_state_ctx_mode", type=str, default="diff")
    p.add_argument("--can_long_state_min_scale", type=int, default=0)
    p.add_argument(
        "--can_long_state_position",
        type=str,
        default="input",
        choices=["input", "after_first", "every_block"],
    )
    p.add_argument(
        "--can_long_aux_mode",
        type=str,
        default="both",
        choices=[
            "trend",
            "stats",
            "both",
            "memory",
            "memory_stats",
            "memory_trend",
            "memory_both",
            "detail",
            "trend_detail",
            "detail_stats",
            "detail_both",
        ],
    )
    # Long-context teacher, short-context student distillation. The student
    # model slices the encoder input internally and only sees the short window.
    p.add_argument("--distill_student_seq_len", type=int, default=96)
    p.add_argument(
        "--can_distill_student_model",
        type=str,
        default="CANTimeImage",
        choices=["CANTimeImage", "CANTimeMixerPP", "CANTimeMixerPPV2"],
    )
    p.add_argument("--distill_teacher_model", type=str, default="CANPatchTST")
    p.add_argument("--distill_teacher_checkpoint", type=str, default="")
    p.add_argument("--distill_alpha", type=float, default=0.0)
    p.add_argument("--distill_teacher_seq_len", type=int, default=192)
    p.add_argument("--distill_teacher_label_len", type=int, default=48)
    p.add_argument("--distill_teacher_d_model", type=int, default=128)
    p.add_argument("--distill_teacher_e_layers", type=int, default=2)
    p.add_argument("--distill_teacher_d_ff", type=int, default=192)
    p.add_argument("--distill_teacher_patch_len", type=int, default=16)
    p.add_argument("--distill_teacher_can_stride", type=int, default=8)
    p.add_argument("--distill_teacher_can_shifts", type=str, default="1,2,4,8,16")
    p.add_argument("--distill_teacher_can_cli_mode", type=str, default="full")
    p.add_argument("--distill_teacher_can_temporal_cli_mode", type=str, default="full")
    p.add_argument("--distill_teacher_can_temporal_roll", type=int, default=1)
    p.add_argument("--distill_teacher_can_context_pyramid", type=int, default=1)
    p.add_argument("--distill_teacher_can_use_gffng", type=int, default=1)
    p.add_argument("--distill_teacher_can_use_orth", type=int, default=0)
    p.add_argument("--distill_teacher_can_ctx_mode", type=str, default="diff")
    p.add_argument("--distill_teacher_can_drop_path", type=float, default=0.05)
    p.add_argument("--distill_teacher_can_kernel_size", type=int, default=3)
    p.add_argument("--distill_teacher_can_init_values", type=float, default=1e-4)
    # training
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--itr", type=int, default=1)
    p.add_argument("--train_epochs", type=int, default=2)
    p.add_argument(
        "--max_train_steps",
        type=int,
        default=0,
        help="stop after this many optimizer steps; 0 disables step budgeting",
    )
    p.add_argument(
        "--stop_after_epochs",
        type=int,
        default=0,
        help="stop after completing this many train/vali/test epochs; 0 disables",
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_epochs", type=int, default=1)
    p.add_argument(
        "--weight_averaging",
        type=str,
        default="none",
        choices=["none", "ema"]
    )
    p.add_argument("--ema_decay", type=float, default=0.995)
    p.add_argument("--ema_start_epoch", type=int, default=1)
    p.add_argument("--loss_horizon_weight_start", type=int, default=0)
    p.add_argument("--loss_horizon_weight", type=float, default=1.0)
    p.add_argument(
        "--loss_horizon_weight_mode",
        type=str,
        default="step",
        choices=["step", "ramp"],
    )
    p.add_argument("--loss_volatility_weight", type=float, default=0.0)
    p.add_argument("--loss_level_weight", type=float, default=0.0)
    p.add_argument("--loss_variance_weight", type=float, default=0.0)
    p.add_argument("--loss_variable_weights", type=str, default="")
    p.add_argument("--loss_range_weight", type=float, default=0.0)
    p.add_argument("--loss_tail_bias_weight", type=float, default=0.0)
    p.add_argument("--loss_tail_bias_start", type=int, default=0)
    p.add_argument("--loss_tail_bias_variables", type=str, default="")
    p.add_argument("--loss_tail_level_weight", type=float, default=0.0)
    p.add_argument("--loss_tail_level_start", type=int, default=0)
    p.add_argument("--loss_tail_level_variables", type=str, default="")
    p.add_argument("--loss_tail_hard_weight", type=float, default=0.0)
    p.add_argument("--loss_tail_hard_start", type=int, default=0)
    p.add_argument("--loss_tail_hard_power", type=float, default=1.0)
    p.add_argument("--loss_tail_hard_clip", type=float, default=3.0)
    p.add_argument("--loss_tail_lowpass_weight", type=float, default=0.0)
    p.add_argument("--loss_tail_lowpass_start", type=int, default=0)
    p.add_argument("--loss_tail_lowpass_kernel", type=int, default=9)
    p.add_argument("--loss_tail_lowpass_variables", type=str, default="")
    p.add_argument("--loss_consistency_weight", type=float, default=0.0)
    p.add_argument("--loss_consistency_start", type=int, default=0)
    p.add_argument("--loss_consistency_variables", type=str, default="")
    p.add_argument("--des", type=str, default="CAN_RELEASE")
    p.add_argument("--loss", type=str, default="MSE")
    p.add_argument("--lradj", type=str, default="cosine")
    p.add_argument("--use_amp", action="store_true", default=False)
    p.add_argument("--validation_only", type=int, default=0)
    p.add_argument(
        "--vali_metric_mode",
        type=str,
        default="all",
        choices=["all", "tail", "weighted"],
    )
    p.add_argument("--vali_metric_horizon_start", type=int, default=0)
    p.add_argument("--save_epoch_checkpoints", type=int, default=0)
    p.add_argument("--save_epoch_start", type=int, default=1)
    p.add_argument("--use_dtw", action="store_true", default=False)
    p.add_argument("--use_gpu", action="store_true", default=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--gpu_type", type=str, default="cuda")
    p.add_argument("--use_multi_gpu", action="store_true", default=False)
    p.add_argument("--devices", type=str, default="0")
    p.add_argument("--p_hidden_dims", type=int, nargs="+", default=[128, 128])
    p.add_argument("--p_hidden_layers", type=int, default=2)
    p.add_argument("--seed", type=int, default=2)
    return p


def _setting(args, ii):
    return (f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_"
            f"ft{args.features}_sl{args.seq_len}_ll{args.label_len}_"
            f"pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_"
            f"el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_"
            f"expand{args.expand}_dc{args.d_conv}_fc{args.factor}_"
            f"eb{args.embed}_dt{args.distil}_{args.des}_{ii}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device(f"cuda:{args.gpu}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and args.use_gpu:
        args.device = torch.device("mps")
    else:
        args.device = torch.device("cpu")

    if args.use_gpu and args.use_multi_gpu:
        args.device_ids = [int(x) for x in args.devices.replace(" ", "").split(",")]
        args.gpu = args.device_ids[0]

    print_args(args)
    Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            s = _setting(args, ii)
            print(f">>>>>>>start training : {s}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            exp.train(s)
            if not args.validation_only:
                print(f">>>>>>>testing : {s}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                exp.test(s)
            torch.cuda.empty_cache() if args.device.type == "cuda" else None
    else:
        exp = Exp(args)
        s = _setting(args, 0)
        if args.test_checkpoint:
            path = os.path.join(args.checkpoints, s)
            os.makedirs(path, exist_ok=True)
            import shutil
            shutil.copy2(args.test_checkpoint, os.path.join(path, "checkpoint.pth"))
            args.strict_checkpoint = int(args.strict_checkpoint)
        print(f">>>>>>>testing : {s}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(s, test=1)
