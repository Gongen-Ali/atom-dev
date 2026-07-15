from __future__ import annotations
import torch
class GraphModule(torch.nn.Module):
    def forward(self, s72: "Sym(s72)", L_input_ids_: "i32[s72]", L_self_modules_embed_parameters_weight_: "bf16[32320, 4096]", L_self_modules_layers_modules_0_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_0_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_0_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_0_modules_attn_norm_parameters_weight_: "bf16[4096]", s80: "Sym(s80)", L_positions_: "i64[s80]", L_self_modules_layers_modules_0_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_0_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_0_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_0_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_0_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_1_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_1_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_1_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_1_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_1_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_1_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_1_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_1_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_1_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_2_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_2_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_2_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_2_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_2_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_2_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_2_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_2_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_2_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_3_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_3_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_3_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_3_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_3_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_3_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_3_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_3_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_3_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_4_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_4_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_4_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_4_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_4_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_4_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_4_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_4_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_4_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_5_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_5_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_5_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_5_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_5_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_5_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_5_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_5_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_5_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_6_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_6_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_6_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_6_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_6_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_6_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_6_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_6_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_6_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_7_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_7_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_7_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_7_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_7_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_7_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_7_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_7_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_7_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_8_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_8_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_8_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_8_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_8_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_8_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_8_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_8_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_8_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_9_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_9_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_9_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_9_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_9_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_9_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_9_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_9_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_9_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_10_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_10_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_10_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_10_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_10_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_10_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_10_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_10_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_10_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_11_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_11_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_11_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_11_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_11_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_11_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_11_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_11_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_11_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_12_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_12_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_12_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_12_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_12_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_12_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_12_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_12_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_12_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_13_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_13_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_13_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_13_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_13_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_13_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_13_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_13_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_13_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_14_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_14_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_14_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_14_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_14_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_14_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_14_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_14_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_14_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_15_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_15_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_15_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_15_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_15_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_15_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_15_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_15_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_15_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_16_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_16_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_16_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_16_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_16_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_16_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_16_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_16_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_16_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_17_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_17_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_17_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_17_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_17_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_17_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_17_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_17_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_17_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_18_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_18_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_18_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_18_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_18_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_18_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_18_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_18_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_18_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_19_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_19_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_19_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_19_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_19_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_19_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_19_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_19_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_19_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_20_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_20_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_20_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_20_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_20_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_20_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_20_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_20_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_20_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_21_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_21_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_21_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_21_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_21_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_21_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_21_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_21_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_21_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_22_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_22_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_22_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_22_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_22_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_22_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_22_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_22_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_22_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_23_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_23_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_23_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_23_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_23_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_23_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_23_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_23_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_23_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_24_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_24_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_24_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_24_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_24_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_24_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_24_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_24_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_24_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_25_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_25_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_25_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_25_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_25_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_25_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_25_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_25_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_25_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_26_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_26_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_26_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_26_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_26_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_26_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_26_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_26_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_26_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_27_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_27_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_27_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_27_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_27_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_27_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_27_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_27_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_27_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_28_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_28_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_28_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_28_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_28_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_28_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_28_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_28_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_28_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_29_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_29_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_29_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_29_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_29_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_29_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_29_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_29_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_29_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_30_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_30_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_30_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_30_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_30_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_30_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_30_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_30_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_30_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_31_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_31_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_31_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_31_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_31_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_31_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_31_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_31_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_31_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_32_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_32_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_32_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_32_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_32_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_32_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_32_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_32_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_32_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_33_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_33_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_33_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_33_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_33_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_33_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_33_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_33_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_33_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_34_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_34_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_34_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_34_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_34_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_34_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_34_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_34_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_34_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_35_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_35_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_35_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_35_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_35_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_35_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_35_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_35_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_35_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_36_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_36_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_36_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_36_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_36_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_36_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_36_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_36_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_36_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_37_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_37_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_37_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_37_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_37_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_37_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_37_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_37_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_37_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_38_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_38_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_38_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_38_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_38_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_38_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_38_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_38_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_38_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_39_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_39_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_39_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_39_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_39_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_39_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_39_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_39_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_39_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_40_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_40_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_40_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_40_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_40_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_40_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_40_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_40_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_40_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_41_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_41_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_41_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_41_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_41_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_41_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_41_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_41_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_41_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", L_self_modules_layers_modules_42_parameters_hc_attn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_42_parameters_hc_attn_scale_: "f32[3]", L_self_modules_layers_modules_42_parameters_hc_attn_base_: "f32[24]", L_self_modules_layers_modules_42_modules_attn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_42_parameters_hc_ffn_fn_: "f32[24, 16384]", L_self_modules_layers_modules_42_parameters_hc_ffn_scale_: "f32[3]", L_self_modules_layers_modules_42_parameters_hc_ffn_base_: "f32[24]", L_self_modules_layers_modules_42_modules_ffn_norm_parameters_weight_: "bf16[4096]", L_self_modules_layers_modules_42_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]"):
        l_input_ids_ = L_input_ids_
        l_self_modules_embed_parameters_weight_ = L_self_modules_embed_parameters_weight_
        l_self_modules_layers_modules_0_parameters_hc_attn_fn_ = L_self_modules_layers_modules_0_parameters_hc_attn_fn_
        l_self_modules_layers_modules_0_parameters_hc_attn_scale_ = L_self_modules_layers_modules_0_parameters_hc_attn_scale_
        l_self_modules_layers_modules_0_parameters_hc_attn_base_ = L_self_modules_layers_modules_0_parameters_hc_attn_base_
        l_self_modules_layers_modules_0_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_0_modules_attn_norm_parameters_weight_
        l_positions_ = L_positions_
        l_self_modules_layers_modules_0_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_0_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_0_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_0_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_0_parameters_hc_ffn_base_ = L_self_modules_layers_modules_0_parameters_hc_ffn_base_
        l_self_modules_layers_modules_0_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_0_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_0_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_0_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_1_parameters_hc_attn_fn_ = L_self_modules_layers_modules_1_parameters_hc_attn_fn_
        l_self_modules_layers_modules_1_parameters_hc_attn_scale_ = L_self_modules_layers_modules_1_parameters_hc_attn_scale_
        l_self_modules_layers_modules_1_parameters_hc_attn_base_ = L_self_modules_layers_modules_1_parameters_hc_attn_base_
        l_self_modules_layers_modules_1_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_1_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_1_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_1_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_1_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_1_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_1_parameters_hc_ffn_base_ = L_self_modules_layers_modules_1_parameters_hc_ffn_base_
        l_self_modules_layers_modules_1_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_1_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_1_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_1_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_2_parameters_hc_attn_fn_ = L_self_modules_layers_modules_2_parameters_hc_attn_fn_
        l_self_modules_layers_modules_2_parameters_hc_attn_scale_ = L_self_modules_layers_modules_2_parameters_hc_attn_scale_
        l_self_modules_layers_modules_2_parameters_hc_attn_base_ = L_self_modules_layers_modules_2_parameters_hc_attn_base_
        l_self_modules_layers_modules_2_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_2_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_2_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_2_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_2_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_2_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_2_parameters_hc_ffn_base_ = L_self_modules_layers_modules_2_parameters_hc_ffn_base_
        l_self_modules_layers_modules_2_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_2_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_2_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_2_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_3_parameters_hc_attn_fn_ = L_self_modules_layers_modules_3_parameters_hc_attn_fn_
        l_self_modules_layers_modules_3_parameters_hc_attn_scale_ = L_self_modules_layers_modules_3_parameters_hc_attn_scale_
        l_self_modules_layers_modules_3_parameters_hc_attn_base_ = L_self_modules_layers_modules_3_parameters_hc_attn_base_
        l_self_modules_layers_modules_3_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_3_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_3_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_3_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_3_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_3_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_3_parameters_hc_ffn_base_ = L_self_modules_layers_modules_3_parameters_hc_ffn_base_
        l_self_modules_layers_modules_3_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_3_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_3_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_3_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_4_parameters_hc_attn_fn_ = L_self_modules_layers_modules_4_parameters_hc_attn_fn_
        l_self_modules_layers_modules_4_parameters_hc_attn_scale_ = L_self_modules_layers_modules_4_parameters_hc_attn_scale_
        l_self_modules_layers_modules_4_parameters_hc_attn_base_ = L_self_modules_layers_modules_4_parameters_hc_attn_base_
        l_self_modules_layers_modules_4_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_4_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_4_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_4_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_4_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_4_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_4_parameters_hc_ffn_base_ = L_self_modules_layers_modules_4_parameters_hc_ffn_base_
        l_self_modules_layers_modules_4_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_4_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_4_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_4_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_5_parameters_hc_attn_fn_ = L_self_modules_layers_modules_5_parameters_hc_attn_fn_
        l_self_modules_layers_modules_5_parameters_hc_attn_scale_ = L_self_modules_layers_modules_5_parameters_hc_attn_scale_
        l_self_modules_layers_modules_5_parameters_hc_attn_base_ = L_self_modules_layers_modules_5_parameters_hc_attn_base_
        l_self_modules_layers_modules_5_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_5_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_5_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_5_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_5_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_5_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_5_parameters_hc_ffn_base_ = L_self_modules_layers_modules_5_parameters_hc_ffn_base_
        l_self_modules_layers_modules_5_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_5_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_5_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_5_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_6_parameters_hc_attn_fn_ = L_self_modules_layers_modules_6_parameters_hc_attn_fn_
        l_self_modules_layers_modules_6_parameters_hc_attn_scale_ = L_self_modules_layers_modules_6_parameters_hc_attn_scale_
        l_self_modules_layers_modules_6_parameters_hc_attn_base_ = L_self_modules_layers_modules_6_parameters_hc_attn_base_
        l_self_modules_layers_modules_6_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_6_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_6_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_6_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_6_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_6_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_6_parameters_hc_ffn_base_ = L_self_modules_layers_modules_6_parameters_hc_ffn_base_
        l_self_modules_layers_modules_6_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_6_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_6_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_6_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_7_parameters_hc_attn_fn_ = L_self_modules_layers_modules_7_parameters_hc_attn_fn_
        l_self_modules_layers_modules_7_parameters_hc_attn_scale_ = L_self_modules_layers_modules_7_parameters_hc_attn_scale_
        l_self_modules_layers_modules_7_parameters_hc_attn_base_ = L_self_modules_layers_modules_7_parameters_hc_attn_base_
        l_self_modules_layers_modules_7_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_7_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_7_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_7_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_7_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_7_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_7_parameters_hc_ffn_base_ = L_self_modules_layers_modules_7_parameters_hc_ffn_base_
        l_self_modules_layers_modules_7_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_7_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_7_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_7_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_8_parameters_hc_attn_fn_ = L_self_modules_layers_modules_8_parameters_hc_attn_fn_
        l_self_modules_layers_modules_8_parameters_hc_attn_scale_ = L_self_modules_layers_modules_8_parameters_hc_attn_scale_
        l_self_modules_layers_modules_8_parameters_hc_attn_base_ = L_self_modules_layers_modules_8_parameters_hc_attn_base_
        l_self_modules_layers_modules_8_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_8_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_8_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_8_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_8_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_8_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_8_parameters_hc_ffn_base_ = L_self_modules_layers_modules_8_parameters_hc_ffn_base_
        l_self_modules_layers_modules_8_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_8_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_8_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_8_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_9_parameters_hc_attn_fn_ = L_self_modules_layers_modules_9_parameters_hc_attn_fn_
        l_self_modules_layers_modules_9_parameters_hc_attn_scale_ = L_self_modules_layers_modules_9_parameters_hc_attn_scale_
        l_self_modules_layers_modules_9_parameters_hc_attn_base_ = L_self_modules_layers_modules_9_parameters_hc_attn_base_
        l_self_modules_layers_modules_9_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_9_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_9_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_9_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_9_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_9_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_9_parameters_hc_ffn_base_ = L_self_modules_layers_modules_9_parameters_hc_ffn_base_
        l_self_modules_layers_modules_9_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_9_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_9_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_9_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_10_parameters_hc_attn_fn_ = L_self_modules_layers_modules_10_parameters_hc_attn_fn_
        l_self_modules_layers_modules_10_parameters_hc_attn_scale_ = L_self_modules_layers_modules_10_parameters_hc_attn_scale_
        l_self_modules_layers_modules_10_parameters_hc_attn_base_ = L_self_modules_layers_modules_10_parameters_hc_attn_base_
        l_self_modules_layers_modules_10_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_10_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_10_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_10_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_10_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_10_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_10_parameters_hc_ffn_base_ = L_self_modules_layers_modules_10_parameters_hc_ffn_base_
        l_self_modules_layers_modules_10_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_10_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_10_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_10_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_11_parameters_hc_attn_fn_ = L_self_modules_layers_modules_11_parameters_hc_attn_fn_
        l_self_modules_layers_modules_11_parameters_hc_attn_scale_ = L_self_modules_layers_modules_11_parameters_hc_attn_scale_
        l_self_modules_layers_modules_11_parameters_hc_attn_base_ = L_self_modules_layers_modules_11_parameters_hc_attn_base_
        l_self_modules_layers_modules_11_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_11_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_11_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_11_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_11_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_11_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_11_parameters_hc_ffn_base_ = L_self_modules_layers_modules_11_parameters_hc_ffn_base_
        l_self_modules_layers_modules_11_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_11_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_11_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_11_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_12_parameters_hc_attn_fn_ = L_self_modules_layers_modules_12_parameters_hc_attn_fn_
        l_self_modules_layers_modules_12_parameters_hc_attn_scale_ = L_self_modules_layers_modules_12_parameters_hc_attn_scale_
        l_self_modules_layers_modules_12_parameters_hc_attn_base_ = L_self_modules_layers_modules_12_parameters_hc_attn_base_
        l_self_modules_layers_modules_12_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_12_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_12_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_12_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_12_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_12_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_12_parameters_hc_ffn_base_ = L_self_modules_layers_modules_12_parameters_hc_ffn_base_
        l_self_modules_layers_modules_12_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_12_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_12_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_12_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_13_parameters_hc_attn_fn_ = L_self_modules_layers_modules_13_parameters_hc_attn_fn_
        l_self_modules_layers_modules_13_parameters_hc_attn_scale_ = L_self_modules_layers_modules_13_parameters_hc_attn_scale_
        l_self_modules_layers_modules_13_parameters_hc_attn_base_ = L_self_modules_layers_modules_13_parameters_hc_attn_base_
        l_self_modules_layers_modules_13_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_13_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_13_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_13_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_13_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_13_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_13_parameters_hc_ffn_base_ = L_self_modules_layers_modules_13_parameters_hc_ffn_base_
        l_self_modules_layers_modules_13_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_13_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_13_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_13_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_14_parameters_hc_attn_fn_ = L_self_modules_layers_modules_14_parameters_hc_attn_fn_
        l_self_modules_layers_modules_14_parameters_hc_attn_scale_ = L_self_modules_layers_modules_14_parameters_hc_attn_scale_
        l_self_modules_layers_modules_14_parameters_hc_attn_base_ = L_self_modules_layers_modules_14_parameters_hc_attn_base_
        l_self_modules_layers_modules_14_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_14_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_14_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_14_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_14_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_14_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_14_parameters_hc_ffn_base_ = L_self_modules_layers_modules_14_parameters_hc_ffn_base_
        l_self_modules_layers_modules_14_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_14_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_14_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_14_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_15_parameters_hc_attn_fn_ = L_self_modules_layers_modules_15_parameters_hc_attn_fn_
        l_self_modules_layers_modules_15_parameters_hc_attn_scale_ = L_self_modules_layers_modules_15_parameters_hc_attn_scale_
        l_self_modules_layers_modules_15_parameters_hc_attn_base_ = L_self_modules_layers_modules_15_parameters_hc_attn_base_
        l_self_modules_layers_modules_15_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_15_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_15_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_15_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_15_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_15_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_15_parameters_hc_ffn_base_ = L_self_modules_layers_modules_15_parameters_hc_ffn_base_
        l_self_modules_layers_modules_15_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_15_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_15_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_15_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_16_parameters_hc_attn_fn_ = L_self_modules_layers_modules_16_parameters_hc_attn_fn_
        l_self_modules_layers_modules_16_parameters_hc_attn_scale_ = L_self_modules_layers_modules_16_parameters_hc_attn_scale_
        l_self_modules_layers_modules_16_parameters_hc_attn_base_ = L_self_modules_layers_modules_16_parameters_hc_attn_base_
        l_self_modules_layers_modules_16_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_16_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_16_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_16_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_16_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_16_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_16_parameters_hc_ffn_base_ = L_self_modules_layers_modules_16_parameters_hc_ffn_base_
        l_self_modules_layers_modules_16_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_16_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_16_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_16_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_17_parameters_hc_attn_fn_ = L_self_modules_layers_modules_17_parameters_hc_attn_fn_
        l_self_modules_layers_modules_17_parameters_hc_attn_scale_ = L_self_modules_layers_modules_17_parameters_hc_attn_scale_
        l_self_modules_layers_modules_17_parameters_hc_attn_base_ = L_self_modules_layers_modules_17_parameters_hc_attn_base_
        l_self_modules_layers_modules_17_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_17_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_17_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_17_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_17_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_17_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_17_parameters_hc_ffn_base_ = L_self_modules_layers_modules_17_parameters_hc_ffn_base_
        l_self_modules_layers_modules_17_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_17_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_17_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_17_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_18_parameters_hc_attn_fn_ = L_self_modules_layers_modules_18_parameters_hc_attn_fn_
        l_self_modules_layers_modules_18_parameters_hc_attn_scale_ = L_self_modules_layers_modules_18_parameters_hc_attn_scale_
        l_self_modules_layers_modules_18_parameters_hc_attn_base_ = L_self_modules_layers_modules_18_parameters_hc_attn_base_
        l_self_modules_layers_modules_18_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_18_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_18_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_18_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_18_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_18_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_18_parameters_hc_ffn_base_ = L_self_modules_layers_modules_18_parameters_hc_ffn_base_
        l_self_modules_layers_modules_18_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_18_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_18_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_18_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_19_parameters_hc_attn_fn_ = L_self_modules_layers_modules_19_parameters_hc_attn_fn_
        l_self_modules_layers_modules_19_parameters_hc_attn_scale_ = L_self_modules_layers_modules_19_parameters_hc_attn_scale_
        l_self_modules_layers_modules_19_parameters_hc_attn_base_ = L_self_modules_layers_modules_19_parameters_hc_attn_base_
        l_self_modules_layers_modules_19_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_19_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_19_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_19_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_19_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_19_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_19_parameters_hc_ffn_base_ = L_self_modules_layers_modules_19_parameters_hc_ffn_base_
        l_self_modules_layers_modules_19_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_19_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_19_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_19_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_20_parameters_hc_attn_fn_ = L_self_modules_layers_modules_20_parameters_hc_attn_fn_
        l_self_modules_layers_modules_20_parameters_hc_attn_scale_ = L_self_modules_layers_modules_20_parameters_hc_attn_scale_
        l_self_modules_layers_modules_20_parameters_hc_attn_base_ = L_self_modules_layers_modules_20_parameters_hc_attn_base_
        l_self_modules_layers_modules_20_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_20_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_20_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_20_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_20_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_20_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_20_parameters_hc_ffn_base_ = L_self_modules_layers_modules_20_parameters_hc_ffn_base_
        l_self_modules_layers_modules_20_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_20_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_20_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_20_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_21_parameters_hc_attn_fn_ = L_self_modules_layers_modules_21_parameters_hc_attn_fn_
        l_self_modules_layers_modules_21_parameters_hc_attn_scale_ = L_self_modules_layers_modules_21_parameters_hc_attn_scale_
        l_self_modules_layers_modules_21_parameters_hc_attn_base_ = L_self_modules_layers_modules_21_parameters_hc_attn_base_
        l_self_modules_layers_modules_21_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_21_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_21_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_21_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_21_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_21_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_21_parameters_hc_ffn_base_ = L_self_modules_layers_modules_21_parameters_hc_ffn_base_
        l_self_modules_layers_modules_21_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_21_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_21_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_21_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_22_parameters_hc_attn_fn_ = L_self_modules_layers_modules_22_parameters_hc_attn_fn_
        l_self_modules_layers_modules_22_parameters_hc_attn_scale_ = L_self_modules_layers_modules_22_parameters_hc_attn_scale_
        l_self_modules_layers_modules_22_parameters_hc_attn_base_ = L_self_modules_layers_modules_22_parameters_hc_attn_base_
        l_self_modules_layers_modules_22_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_22_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_22_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_22_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_22_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_22_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_22_parameters_hc_ffn_base_ = L_self_modules_layers_modules_22_parameters_hc_ffn_base_
        l_self_modules_layers_modules_22_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_22_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_22_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_22_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_23_parameters_hc_attn_fn_ = L_self_modules_layers_modules_23_parameters_hc_attn_fn_
        l_self_modules_layers_modules_23_parameters_hc_attn_scale_ = L_self_modules_layers_modules_23_parameters_hc_attn_scale_
        l_self_modules_layers_modules_23_parameters_hc_attn_base_ = L_self_modules_layers_modules_23_parameters_hc_attn_base_
        l_self_modules_layers_modules_23_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_23_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_23_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_23_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_23_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_23_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_23_parameters_hc_ffn_base_ = L_self_modules_layers_modules_23_parameters_hc_ffn_base_
        l_self_modules_layers_modules_23_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_23_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_23_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_23_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_24_parameters_hc_attn_fn_ = L_self_modules_layers_modules_24_parameters_hc_attn_fn_
        l_self_modules_layers_modules_24_parameters_hc_attn_scale_ = L_self_modules_layers_modules_24_parameters_hc_attn_scale_
        l_self_modules_layers_modules_24_parameters_hc_attn_base_ = L_self_modules_layers_modules_24_parameters_hc_attn_base_
        l_self_modules_layers_modules_24_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_24_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_24_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_24_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_24_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_24_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_24_parameters_hc_ffn_base_ = L_self_modules_layers_modules_24_parameters_hc_ffn_base_
        l_self_modules_layers_modules_24_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_24_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_24_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_24_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_25_parameters_hc_attn_fn_ = L_self_modules_layers_modules_25_parameters_hc_attn_fn_
        l_self_modules_layers_modules_25_parameters_hc_attn_scale_ = L_self_modules_layers_modules_25_parameters_hc_attn_scale_
        l_self_modules_layers_modules_25_parameters_hc_attn_base_ = L_self_modules_layers_modules_25_parameters_hc_attn_base_
        l_self_modules_layers_modules_25_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_25_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_25_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_25_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_25_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_25_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_25_parameters_hc_ffn_base_ = L_self_modules_layers_modules_25_parameters_hc_ffn_base_
        l_self_modules_layers_modules_25_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_25_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_25_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_25_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_26_parameters_hc_attn_fn_ = L_self_modules_layers_modules_26_parameters_hc_attn_fn_
        l_self_modules_layers_modules_26_parameters_hc_attn_scale_ = L_self_modules_layers_modules_26_parameters_hc_attn_scale_
        l_self_modules_layers_modules_26_parameters_hc_attn_base_ = L_self_modules_layers_modules_26_parameters_hc_attn_base_
        l_self_modules_layers_modules_26_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_26_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_26_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_26_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_26_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_26_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_26_parameters_hc_ffn_base_ = L_self_modules_layers_modules_26_parameters_hc_ffn_base_
        l_self_modules_layers_modules_26_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_26_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_26_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_26_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_27_parameters_hc_attn_fn_ = L_self_modules_layers_modules_27_parameters_hc_attn_fn_
        l_self_modules_layers_modules_27_parameters_hc_attn_scale_ = L_self_modules_layers_modules_27_parameters_hc_attn_scale_
        l_self_modules_layers_modules_27_parameters_hc_attn_base_ = L_self_modules_layers_modules_27_parameters_hc_attn_base_
        l_self_modules_layers_modules_27_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_27_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_27_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_27_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_27_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_27_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_27_parameters_hc_ffn_base_ = L_self_modules_layers_modules_27_parameters_hc_ffn_base_
        l_self_modules_layers_modules_27_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_27_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_27_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_27_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_28_parameters_hc_attn_fn_ = L_self_modules_layers_modules_28_parameters_hc_attn_fn_
        l_self_modules_layers_modules_28_parameters_hc_attn_scale_ = L_self_modules_layers_modules_28_parameters_hc_attn_scale_
        l_self_modules_layers_modules_28_parameters_hc_attn_base_ = L_self_modules_layers_modules_28_parameters_hc_attn_base_
        l_self_modules_layers_modules_28_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_28_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_28_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_28_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_28_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_28_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_28_parameters_hc_ffn_base_ = L_self_modules_layers_modules_28_parameters_hc_ffn_base_
        l_self_modules_layers_modules_28_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_28_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_28_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_28_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_29_parameters_hc_attn_fn_ = L_self_modules_layers_modules_29_parameters_hc_attn_fn_
        l_self_modules_layers_modules_29_parameters_hc_attn_scale_ = L_self_modules_layers_modules_29_parameters_hc_attn_scale_
        l_self_modules_layers_modules_29_parameters_hc_attn_base_ = L_self_modules_layers_modules_29_parameters_hc_attn_base_
        l_self_modules_layers_modules_29_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_29_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_29_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_29_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_29_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_29_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_29_parameters_hc_ffn_base_ = L_self_modules_layers_modules_29_parameters_hc_ffn_base_
        l_self_modules_layers_modules_29_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_29_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_29_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_29_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_30_parameters_hc_attn_fn_ = L_self_modules_layers_modules_30_parameters_hc_attn_fn_
        l_self_modules_layers_modules_30_parameters_hc_attn_scale_ = L_self_modules_layers_modules_30_parameters_hc_attn_scale_
        l_self_modules_layers_modules_30_parameters_hc_attn_base_ = L_self_modules_layers_modules_30_parameters_hc_attn_base_
        l_self_modules_layers_modules_30_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_30_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_30_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_30_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_30_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_30_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_30_parameters_hc_ffn_base_ = L_self_modules_layers_modules_30_parameters_hc_ffn_base_
        l_self_modules_layers_modules_30_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_30_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_30_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_30_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_31_parameters_hc_attn_fn_ = L_self_modules_layers_modules_31_parameters_hc_attn_fn_
        l_self_modules_layers_modules_31_parameters_hc_attn_scale_ = L_self_modules_layers_modules_31_parameters_hc_attn_scale_
        l_self_modules_layers_modules_31_parameters_hc_attn_base_ = L_self_modules_layers_modules_31_parameters_hc_attn_base_
        l_self_modules_layers_modules_31_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_31_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_31_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_31_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_31_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_31_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_31_parameters_hc_ffn_base_ = L_self_modules_layers_modules_31_parameters_hc_ffn_base_
        l_self_modules_layers_modules_31_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_31_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_31_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_31_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_32_parameters_hc_attn_fn_ = L_self_modules_layers_modules_32_parameters_hc_attn_fn_
        l_self_modules_layers_modules_32_parameters_hc_attn_scale_ = L_self_modules_layers_modules_32_parameters_hc_attn_scale_
        l_self_modules_layers_modules_32_parameters_hc_attn_base_ = L_self_modules_layers_modules_32_parameters_hc_attn_base_
        l_self_modules_layers_modules_32_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_32_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_32_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_32_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_32_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_32_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_32_parameters_hc_ffn_base_ = L_self_modules_layers_modules_32_parameters_hc_ffn_base_
        l_self_modules_layers_modules_32_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_32_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_32_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_32_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_33_parameters_hc_attn_fn_ = L_self_modules_layers_modules_33_parameters_hc_attn_fn_
        l_self_modules_layers_modules_33_parameters_hc_attn_scale_ = L_self_modules_layers_modules_33_parameters_hc_attn_scale_
        l_self_modules_layers_modules_33_parameters_hc_attn_base_ = L_self_modules_layers_modules_33_parameters_hc_attn_base_
        l_self_modules_layers_modules_33_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_33_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_33_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_33_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_33_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_33_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_33_parameters_hc_ffn_base_ = L_self_modules_layers_modules_33_parameters_hc_ffn_base_
        l_self_modules_layers_modules_33_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_33_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_33_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_33_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_34_parameters_hc_attn_fn_ = L_self_modules_layers_modules_34_parameters_hc_attn_fn_
        l_self_modules_layers_modules_34_parameters_hc_attn_scale_ = L_self_modules_layers_modules_34_parameters_hc_attn_scale_
        l_self_modules_layers_modules_34_parameters_hc_attn_base_ = L_self_modules_layers_modules_34_parameters_hc_attn_base_
        l_self_modules_layers_modules_34_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_34_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_34_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_34_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_34_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_34_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_34_parameters_hc_ffn_base_ = L_self_modules_layers_modules_34_parameters_hc_ffn_base_
        l_self_modules_layers_modules_34_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_34_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_34_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_34_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_35_parameters_hc_attn_fn_ = L_self_modules_layers_modules_35_parameters_hc_attn_fn_
        l_self_modules_layers_modules_35_parameters_hc_attn_scale_ = L_self_modules_layers_modules_35_parameters_hc_attn_scale_
        l_self_modules_layers_modules_35_parameters_hc_attn_base_ = L_self_modules_layers_modules_35_parameters_hc_attn_base_
        l_self_modules_layers_modules_35_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_35_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_35_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_35_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_35_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_35_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_35_parameters_hc_ffn_base_ = L_self_modules_layers_modules_35_parameters_hc_ffn_base_
        l_self_modules_layers_modules_35_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_35_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_35_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_35_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_36_parameters_hc_attn_fn_ = L_self_modules_layers_modules_36_parameters_hc_attn_fn_
        l_self_modules_layers_modules_36_parameters_hc_attn_scale_ = L_self_modules_layers_modules_36_parameters_hc_attn_scale_
        l_self_modules_layers_modules_36_parameters_hc_attn_base_ = L_self_modules_layers_modules_36_parameters_hc_attn_base_
        l_self_modules_layers_modules_36_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_36_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_36_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_36_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_36_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_36_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_36_parameters_hc_ffn_base_ = L_self_modules_layers_modules_36_parameters_hc_ffn_base_
        l_self_modules_layers_modules_36_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_36_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_36_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_36_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_37_parameters_hc_attn_fn_ = L_self_modules_layers_modules_37_parameters_hc_attn_fn_
        l_self_modules_layers_modules_37_parameters_hc_attn_scale_ = L_self_modules_layers_modules_37_parameters_hc_attn_scale_
        l_self_modules_layers_modules_37_parameters_hc_attn_base_ = L_self_modules_layers_modules_37_parameters_hc_attn_base_
        l_self_modules_layers_modules_37_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_37_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_37_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_37_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_37_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_37_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_37_parameters_hc_ffn_base_ = L_self_modules_layers_modules_37_parameters_hc_ffn_base_
        l_self_modules_layers_modules_37_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_37_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_37_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_37_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_38_parameters_hc_attn_fn_ = L_self_modules_layers_modules_38_parameters_hc_attn_fn_
        l_self_modules_layers_modules_38_parameters_hc_attn_scale_ = L_self_modules_layers_modules_38_parameters_hc_attn_scale_
        l_self_modules_layers_modules_38_parameters_hc_attn_base_ = L_self_modules_layers_modules_38_parameters_hc_attn_base_
        l_self_modules_layers_modules_38_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_38_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_38_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_38_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_38_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_38_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_38_parameters_hc_ffn_base_ = L_self_modules_layers_modules_38_parameters_hc_ffn_base_
        l_self_modules_layers_modules_38_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_38_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_38_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_38_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_39_parameters_hc_attn_fn_ = L_self_modules_layers_modules_39_parameters_hc_attn_fn_
        l_self_modules_layers_modules_39_parameters_hc_attn_scale_ = L_self_modules_layers_modules_39_parameters_hc_attn_scale_
        l_self_modules_layers_modules_39_parameters_hc_attn_base_ = L_self_modules_layers_modules_39_parameters_hc_attn_base_
        l_self_modules_layers_modules_39_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_39_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_39_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_39_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_39_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_39_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_39_parameters_hc_ffn_base_ = L_self_modules_layers_modules_39_parameters_hc_ffn_base_
        l_self_modules_layers_modules_39_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_39_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_39_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_39_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_40_parameters_hc_attn_fn_ = L_self_modules_layers_modules_40_parameters_hc_attn_fn_
        l_self_modules_layers_modules_40_parameters_hc_attn_scale_ = L_self_modules_layers_modules_40_parameters_hc_attn_scale_
        l_self_modules_layers_modules_40_parameters_hc_attn_base_ = L_self_modules_layers_modules_40_parameters_hc_attn_base_
        l_self_modules_layers_modules_40_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_40_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_40_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_40_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_40_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_40_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_40_parameters_hc_ffn_base_ = L_self_modules_layers_modules_40_parameters_hc_ffn_base_
        l_self_modules_layers_modules_40_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_40_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_40_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_40_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_41_parameters_hc_attn_fn_ = L_self_modules_layers_modules_41_parameters_hc_attn_fn_
        l_self_modules_layers_modules_41_parameters_hc_attn_scale_ = L_self_modules_layers_modules_41_parameters_hc_attn_scale_
        l_self_modules_layers_modules_41_parameters_hc_attn_base_ = L_self_modules_layers_modules_41_parameters_hc_attn_base_
        l_self_modules_layers_modules_41_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_41_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_41_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_41_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_41_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_41_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_41_parameters_hc_ffn_base_ = L_self_modules_layers_modules_41_parameters_hc_ffn_base_
        l_self_modules_layers_modules_41_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_41_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_41_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_41_modules_ffn_modules_gate_parameters_weight_
        l_self_modules_layers_modules_42_parameters_hc_attn_fn_ = L_self_modules_layers_modules_42_parameters_hc_attn_fn_
        l_self_modules_layers_modules_42_parameters_hc_attn_scale_ = L_self_modules_layers_modules_42_parameters_hc_attn_scale_
        l_self_modules_layers_modules_42_parameters_hc_attn_base_ = L_self_modules_layers_modules_42_parameters_hc_attn_base_
        l_self_modules_layers_modules_42_modules_attn_norm_parameters_weight_ = L_self_modules_layers_modules_42_modules_attn_norm_parameters_weight_
        l_self_modules_layers_modules_42_parameters_hc_ffn_fn_ = L_self_modules_layers_modules_42_parameters_hc_ffn_fn_
        l_self_modules_layers_modules_42_parameters_hc_ffn_scale_ = L_self_modules_layers_modules_42_parameters_hc_ffn_scale_
        l_self_modules_layers_modules_42_parameters_hc_ffn_base_ = L_self_modules_layers_modules_42_parameters_hc_ffn_base_
        l_self_modules_layers_modules_42_modules_ffn_norm_parameters_weight_ = L_self_modules_layers_modules_42_modules_ffn_norm_parameters_weight_
        l_self_modules_layers_modules_42_modules_ffn_modules_gate_parameters_weight_ = L_self_modules_layers_modules_42_modules_ffn_modules_gate_parameters_weight_
        
        # No stacktrace found for following nodes
        submod_0 = self.submod_0(l_input_ids_, s72, l_self_modules_embed_parameters_weight_, l_self_modules_layers_modules_0_parameters_hc_attn_fn_, l_self_modules_layers_modules_0_parameters_hc_attn_scale_, l_self_modules_layers_modules_0_parameters_hc_attn_base_, l_self_modules_layers_modules_0_modules_attn_norm_parameters_weight_);  l_input_ids_ = l_self_modules_embed_parameters_weight_ = l_self_modules_layers_modules_0_parameters_hc_attn_fn_ = l_self_modules_layers_modules_0_parameters_hc_attn_scale_ = l_self_modules_layers_modules_0_parameters_hc_attn_base_ = l_self_modules_layers_modules_0_modules_attn_norm_parameters_weight_ = None
        getitem = submod_0[0]
        getitem_1 = submod_0[1]
        getitem_2 = submod_0[2]
        getitem_3 = submod_0[3];  submod_0 = None
        submod_1 = self.submod_1(getitem, s72, l_positions_, s80);  getitem = None
        submod_2 = self.submod_2(getitem_1, s72, getitem_2, submod_1, getitem_3, l_self_modules_layers_modules_0_parameters_hc_ffn_fn_, l_self_modules_layers_modules_0_parameters_hc_ffn_scale_, l_self_modules_layers_modules_0_parameters_hc_ffn_base_, l_self_modules_layers_modules_0_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_0_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_1_parameters_hc_attn_fn_, l_self_modules_layers_modules_1_parameters_hc_attn_scale_, l_self_modules_layers_modules_1_parameters_hc_attn_base_, l_self_modules_layers_modules_1_modules_attn_norm_parameters_weight_);  getitem_1 = getitem_2 = submod_1 = getitem_3 = l_self_modules_layers_modules_0_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_0_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_0_parameters_hc_ffn_base_ = l_self_modules_layers_modules_0_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_0_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_1_parameters_hc_attn_fn_ = l_self_modules_layers_modules_1_parameters_hc_attn_scale_ = l_self_modules_layers_modules_1_parameters_hc_attn_base_ = l_self_modules_layers_modules_1_modules_attn_norm_parameters_weight_ = None
        getitem_4 = submod_2[0]
        getitem_5 = submod_2[1]
        getitem_6 = submod_2[2]
        getitem_7 = submod_2[3];  submod_2 = None
        submod_3 = self.submod_3(getitem_4, s72, l_positions_, s80);  getitem_4 = None
        submod_4 = self.submod_4(getitem_5, s72, getitem_6, submod_3, getitem_7, l_self_modules_layers_modules_1_parameters_hc_ffn_fn_, l_self_modules_layers_modules_1_parameters_hc_ffn_scale_, l_self_modules_layers_modules_1_parameters_hc_ffn_base_, l_self_modules_layers_modules_1_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_1_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_2_parameters_hc_attn_fn_, l_self_modules_layers_modules_2_parameters_hc_attn_scale_, l_self_modules_layers_modules_2_parameters_hc_attn_base_, l_self_modules_layers_modules_2_modules_attn_norm_parameters_weight_);  getitem_5 = getitem_6 = submod_3 = getitem_7 = l_self_modules_layers_modules_1_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_1_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_1_parameters_hc_ffn_base_ = l_self_modules_layers_modules_1_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_1_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_2_parameters_hc_attn_fn_ = l_self_modules_layers_modules_2_parameters_hc_attn_scale_ = l_self_modules_layers_modules_2_parameters_hc_attn_base_ = l_self_modules_layers_modules_2_modules_attn_norm_parameters_weight_ = None
        getitem_8 = submod_4[0]
        getitem_9 = submod_4[1]
        getitem_10 = submod_4[2]
        getitem_11 = submod_4[3];  submod_4 = None
        submod_5 = self.submod_5(getitem_8, s72, l_positions_, s80);  getitem_8 = None
        submod_6 = self.submod_6(getitem_9, s72, getitem_10, submod_5, getitem_11, l_self_modules_layers_modules_2_parameters_hc_ffn_fn_, l_self_modules_layers_modules_2_parameters_hc_ffn_scale_, l_self_modules_layers_modules_2_parameters_hc_ffn_base_, l_self_modules_layers_modules_2_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_2_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_3_parameters_hc_attn_fn_, l_self_modules_layers_modules_3_parameters_hc_attn_scale_, l_self_modules_layers_modules_3_parameters_hc_attn_base_, l_self_modules_layers_modules_3_modules_attn_norm_parameters_weight_);  getitem_9 = getitem_10 = submod_5 = getitem_11 = l_self_modules_layers_modules_2_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_2_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_2_parameters_hc_ffn_base_ = l_self_modules_layers_modules_2_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_2_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_3_parameters_hc_attn_fn_ = l_self_modules_layers_modules_3_parameters_hc_attn_scale_ = l_self_modules_layers_modules_3_parameters_hc_attn_base_ = l_self_modules_layers_modules_3_modules_attn_norm_parameters_weight_ = None
        getitem_12 = submod_6[0]
        getitem_13 = submod_6[1]
        getitem_14 = submod_6[2]
        getitem_15 = submod_6[3];  submod_6 = None
        submod_7 = self.submod_7(getitem_12, s72, l_positions_, s80);  getitem_12 = None
        submod_8 = self.submod_8(getitem_13, s72, getitem_14, submod_7, getitem_15, l_self_modules_layers_modules_3_parameters_hc_ffn_fn_, l_self_modules_layers_modules_3_parameters_hc_ffn_scale_, l_self_modules_layers_modules_3_parameters_hc_ffn_base_, l_self_modules_layers_modules_3_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_3_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_4_parameters_hc_attn_fn_, l_self_modules_layers_modules_4_parameters_hc_attn_scale_, l_self_modules_layers_modules_4_parameters_hc_attn_base_, l_self_modules_layers_modules_4_modules_attn_norm_parameters_weight_);  getitem_13 = getitem_14 = submod_7 = getitem_15 = l_self_modules_layers_modules_3_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_3_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_3_parameters_hc_ffn_base_ = l_self_modules_layers_modules_3_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_3_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_4_parameters_hc_attn_fn_ = l_self_modules_layers_modules_4_parameters_hc_attn_scale_ = l_self_modules_layers_modules_4_parameters_hc_attn_base_ = l_self_modules_layers_modules_4_modules_attn_norm_parameters_weight_ = None
        getitem_16 = submod_8[0]
        getitem_17 = submod_8[1]
        getitem_18 = submod_8[2]
        getitem_19 = submod_8[3];  submod_8 = None
        submod_9 = self.submod_9(getitem_16, s72, l_positions_, s80);  getitem_16 = None
        submod_10 = self.submod_10(getitem_17, s72, getitem_18, submod_9, getitem_19, l_self_modules_layers_modules_4_parameters_hc_ffn_fn_, l_self_modules_layers_modules_4_parameters_hc_ffn_scale_, l_self_modules_layers_modules_4_parameters_hc_ffn_base_, l_self_modules_layers_modules_4_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_4_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_5_parameters_hc_attn_fn_, l_self_modules_layers_modules_5_parameters_hc_attn_scale_, l_self_modules_layers_modules_5_parameters_hc_attn_base_, l_self_modules_layers_modules_5_modules_attn_norm_parameters_weight_);  getitem_17 = getitem_18 = submod_9 = getitem_19 = l_self_modules_layers_modules_4_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_4_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_4_parameters_hc_ffn_base_ = l_self_modules_layers_modules_4_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_4_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_5_parameters_hc_attn_fn_ = l_self_modules_layers_modules_5_parameters_hc_attn_scale_ = l_self_modules_layers_modules_5_parameters_hc_attn_base_ = l_self_modules_layers_modules_5_modules_attn_norm_parameters_weight_ = None
        getitem_20 = submod_10[0]
        getitem_21 = submod_10[1]
        getitem_22 = submod_10[2]
        getitem_23 = submod_10[3];  submod_10 = None
        submod_11 = self.submod_11(getitem_20, s72, l_positions_, s80);  getitem_20 = None
        submod_12 = self.submod_12(getitem_21, s72, getitem_22, submod_11, getitem_23, l_self_modules_layers_modules_5_parameters_hc_ffn_fn_, l_self_modules_layers_modules_5_parameters_hc_ffn_scale_, l_self_modules_layers_modules_5_parameters_hc_ffn_base_, l_self_modules_layers_modules_5_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_5_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_6_parameters_hc_attn_fn_, l_self_modules_layers_modules_6_parameters_hc_attn_scale_, l_self_modules_layers_modules_6_parameters_hc_attn_base_, l_self_modules_layers_modules_6_modules_attn_norm_parameters_weight_);  getitem_21 = getitem_22 = submod_11 = getitem_23 = l_self_modules_layers_modules_5_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_5_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_5_parameters_hc_ffn_base_ = l_self_modules_layers_modules_5_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_5_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_6_parameters_hc_attn_fn_ = l_self_modules_layers_modules_6_parameters_hc_attn_scale_ = l_self_modules_layers_modules_6_parameters_hc_attn_base_ = l_self_modules_layers_modules_6_modules_attn_norm_parameters_weight_ = None
        getitem_24 = submod_12[0]
        getitem_25 = submod_12[1]
        getitem_26 = submod_12[2]
        getitem_27 = submod_12[3];  submod_12 = None
        submod_13 = self.submod_13(getitem_24, s72, l_positions_, s80);  getitem_24 = None
        submod_14 = self.submod_14(getitem_25, s72, getitem_26, submod_13, getitem_27, l_self_modules_layers_modules_6_parameters_hc_ffn_fn_, l_self_modules_layers_modules_6_parameters_hc_ffn_scale_, l_self_modules_layers_modules_6_parameters_hc_ffn_base_, l_self_modules_layers_modules_6_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_6_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_7_parameters_hc_attn_fn_, l_self_modules_layers_modules_7_parameters_hc_attn_scale_, l_self_modules_layers_modules_7_parameters_hc_attn_base_, l_self_modules_layers_modules_7_modules_attn_norm_parameters_weight_);  getitem_25 = getitem_26 = submod_13 = getitem_27 = l_self_modules_layers_modules_6_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_6_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_6_parameters_hc_ffn_base_ = l_self_modules_layers_modules_6_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_6_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_7_parameters_hc_attn_fn_ = l_self_modules_layers_modules_7_parameters_hc_attn_scale_ = l_self_modules_layers_modules_7_parameters_hc_attn_base_ = l_self_modules_layers_modules_7_modules_attn_norm_parameters_weight_ = None
        getitem_28 = submod_14[0]
        getitem_29 = submod_14[1]
        getitem_30 = submod_14[2]
        getitem_31 = submod_14[3];  submod_14 = None
        submod_15 = self.submod_15(getitem_28, s72, l_positions_, s80);  getitem_28 = None
        submod_16 = self.submod_16(getitem_29, s72, getitem_30, submod_15, getitem_31, l_self_modules_layers_modules_7_parameters_hc_ffn_fn_, l_self_modules_layers_modules_7_parameters_hc_ffn_scale_, l_self_modules_layers_modules_7_parameters_hc_ffn_base_, l_self_modules_layers_modules_7_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_7_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_8_parameters_hc_attn_fn_, l_self_modules_layers_modules_8_parameters_hc_attn_scale_, l_self_modules_layers_modules_8_parameters_hc_attn_base_, l_self_modules_layers_modules_8_modules_attn_norm_parameters_weight_);  getitem_29 = getitem_30 = submod_15 = getitem_31 = l_self_modules_layers_modules_7_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_7_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_7_parameters_hc_ffn_base_ = l_self_modules_layers_modules_7_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_7_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_8_parameters_hc_attn_fn_ = l_self_modules_layers_modules_8_parameters_hc_attn_scale_ = l_self_modules_layers_modules_8_parameters_hc_attn_base_ = l_self_modules_layers_modules_8_modules_attn_norm_parameters_weight_ = None
        getitem_32 = submod_16[0]
        getitem_33 = submod_16[1]
        getitem_34 = submod_16[2]
        getitem_35 = submod_16[3];  submod_16 = None
        submod_17 = self.submod_17(getitem_32, s72, l_positions_, s80);  getitem_32 = None
        submod_18 = self.submod_18(getitem_33, s72, getitem_34, submod_17, getitem_35, l_self_modules_layers_modules_8_parameters_hc_ffn_fn_, l_self_modules_layers_modules_8_parameters_hc_ffn_scale_, l_self_modules_layers_modules_8_parameters_hc_ffn_base_, l_self_modules_layers_modules_8_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_8_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_9_parameters_hc_attn_fn_, l_self_modules_layers_modules_9_parameters_hc_attn_scale_, l_self_modules_layers_modules_9_parameters_hc_attn_base_, l_self_modules_layers_modules_9_modules_attn_norm_parameters_weight_);  getitem_33 = getitem_34 = submod_17 = getitem_35 = l_self_modules_layers_modules_8_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_8_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_8_parameters_hc_ffn_base_ = l_self_modules_layers_modules_8_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_8_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_9_parameters_hc_attn_fn_ = l_self_modules_layers_modules_9_parameters_hc_attn_scale_ = l_self_modules_layers_modules_9_parameters_hc_attn_base_ = l_self_modules_layers_modules_9_modules_attn_norm_parameters_weight_ = None
        getitem_36 = submod_18[0]
        getitem_37 = submod_18[1]
        getitem_38 = submod_18[2]
        getitem_39 = submod_18[3];  submod_18 = None
        submod_19 = self.submod_19(getitem_36, s72, l_positions_, s80);  getitem_36 = None
        submod_20 = self.submod_20(getitem_37, s72, getitem_38, submod_19, getitem_39, l_self_modules_layers_modules_9_parameters_hc_ffn_fn_, l_self_modules_layers_modules_9_parameters_hc_ffn_scale_, l_self_modules_layers_modules_9_parameters_hc_ffn_base_, l_self_modules_layers_modules_9_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_9_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_10_parameters_hc_attn_fn_, l_self_modules_layers_modules_10_parameters_hc_attn_scale_, l_self_modules_layers_modules_10_parameters_hc_attn_base_, l_self_modules_layers_modules_10_modules_attn_norm_parameters_weight_);  getitem_37 = getitem_38 = submod_19 = getitem_39 = l_self_modules_layers_modules_9_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_9_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_9_parameters_hc_ffn_base_ = l_self_modules_layers_modules_9_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_9_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_10_parameters_hc_attn_fn_ = l_self_modules_layers_modules_10_parameters_hc_attn_scale_ = l_self_modules_layers_modules_10_parameters_hc_attn_base_ = l_self_modules_layers_modules_10_modules_attn_norm_parameters_weight_ = None
        getitem_40 = submod_20[0]
        getitem_41 = submod_20[1]
        getitem_42 = submod_20[2]
        getitem_43 = submod_20[3];  submod_20 = None
        submod_21 = self.submod_21(getitem_40, s72, l_positions_, s80);  getitem_40 = None
        submod_22 = self.submod_22(getitem_41, s72, getitem_42, submod_21, getitem_43, l_self_modules_layers_modules_10_parameters_hc_ffn_fn_, l_self_modules_layers_modules_10_parameters_hc_ffn_scale_, l_self_modules_layers_modules_10_parameters_hc_ffn_base_, l_self_modules_layers_modules_10_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_10_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_11_parameters_hc_attn_fn_, l_self_modules_layers_modules_11_parameters_hc_attn_scale_, l_self_modules_layers_modules_11_parameters_hc_attn_base_, l_self_modules_layers_modules_11_modules_attn_norm_parameters_weight_);  getitem_41 = getitem_42 = submod_21 = getitem_43 = l_self_modules_layers_modules_10_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_10_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_10_parameters_hc_ffn_base_ = l_self_modules_layers_modules_10_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_10_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_11_parameters_hc_attn_fn_ = l_self_modules_layers_modules_11_parameters_hc_attn_scale_ = l_self_modules_layers_modules_11_parameters_hc_attn_base_ = l_self_modules_layers_modules_11_modules_attn_norm_parameters_weight_ = None
        getitem_44 = submod_22[0]
        getitem_45 = submod_22[1]
        getitem_46 = submod_22[2]
        getitem_47 = submod_22[3];  submod_22 = None
        submod_23 = self.submod_23(getitem_44, s72, l_positions_, s80);  getitem_44 = None
        submod_24 = self.submod_24(getitem_45, s72, getitem_46, submod_23, getitem_47, l_self_modules_layers_modules_11_parameters_hc_ffn_fn_, l_self_modules_layers_modules_11_parameters_hc_ffn_scale_, l_self_modules_layers_modules_11_parameters_hc_ffn_base_, l_self_modules_layers_modules_11_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_11_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_12_parameters_hc_attn_fn_, l_self_modules_layers_modules_12_parameters_hc_attn_scale_, l_self_modules_layers_modules_12_parameters_hc_attn_base_, l_self_modules_layers_modules_12_modules_attn_norm_parameters_weight_);  getitem_45 = getitem_46 = submod_23 = getitem_47 = l_self_modules_layers_modules_11_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_11_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_11_parameters_hc_ffn_base_ = l_self_modules_layers_modules_11_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_11_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_12_parameters_hc_attn_fn_ = l_self_modules_layers_modules_12_parameters_hc_attn_scale_ = l_self_modules_layers_modules_12_parameters_hc_attn_base_ = l_self_modules_layers_modules_12_modules_attn_norm_parameters_weight_ = None
        getitem_48 = submod_24[0]
        getitem_49 = submod_24[1]
        getitem_50 = submod_24[2]
        getitem_51 = submod_24[3];  submod_24 = None
        submod_25 = self.submod_25(getitem_48, s72, l_positions_, s80);  getitem_48 = None
        submod_26 = self.submod_26(getitem_49, s72, getitem_50, submod_25, getitem_51, l_self_modules_layers_modules_12_parameters_hc_ffn_fn_, l_self_modules_layers_modules_12_parameters_hc_ffn_scale_, l_self_modules_layers_modules_12_parameters_hc_ffn_base_, l_self_modules_layers_modules_12_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_12_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_13_parameters_hc_attn_fn_, l_self_modules_layers_modules_13_parameters_hc_attn_scale_, l_self_modules_layers_modules_13_parameters_hc_attn_base_, l_self_modules_layers_modules_13_modules_attn_norm_parameters_weight_);  getitem_49 = getitem_50 = submod_25 = getitem_51 = l_self_modules_layers_modules_12_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_12_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_12_parameters_hc_ffn_base_ = l_self_modules_layers_modules_12_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_12_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_13_parameters_hc_attn_fn_ = l_self_modules_layers_modules_13_parameters_hc_attn_scale_ = l_self_modules_layers_modules_13_parameters_hc_attn_base_ = l_self_modules_layers_modules_13_modules_attn_norm_parameters_weight_ = None
        getitem_52 = submod_26[0]
        getitem_53 = submod_26[1]
        getitem_54 = submod_26[2]
        getitem_55 = submod_26[3];  submod_26 = None
        submod_27 = self.submod_27(getitem_52, s72, l_positions_, s80);  getitem_52 = None
        submod_28 = self.submod_28(getitem_53, s72, getitem_54, submod_27, getitem_55, l_self_modules_layers_modules_13_parameters_hc_ffn_fn_, l_self_modules_layers_modules_13_parameters_hc_ffn_scale_, l_self_modules_layers_modules_13_parameters_hc_ffn_base_, l_self_modules_layers_modules_13_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_13_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_14_parameters_hc_attn_fn_, l_self_modules_layers_modules_14_parameters_hc_attn_scale_, l_self_modules_layers_modules_14_parameters_hc_attn_base_, l_self_modules_layers_modules_14_modules_attn_norm_parameters_weight_);  getitem_53 = getitem_54 = submod_27 = getitem_55 = l_self_modules_layers_modules_13_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_13_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_13_parameters_hc_ffn_base_ = l_self_modules_layers_modules_13_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_13_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_14_parameters_hc_attn_fn_ = l_self_modules_layers_modules_14_parameters_hc_attn_scale_ = l_self_modules_layers_modules_14_parameters_hc_attn_base_ = l_self_modules_layers_modules_14_modules_attn_norm_parameters_weight_ = None
        getitem_56 = submod_28[0]
        getitem_57 = submod_28[1]
        getitem_58 = submod_28[2]
        getitem_59 = submod_28[3];  submod_28 = None
        submod_29 = self.submod_29(getitem_56, s72, l_positions_, s80);  getitem_56 = None
        submod_30 = self.submod_30(getitem_57, s72, getitem_58, submod_29, getitem_59, l_self_modules_layers_modules_14_parameters_hc_ffn_fn_, l_self_modules_layers_modules_14_parameters_hc_ffn_scale_, l_self_modules_layers_modules_14_parameters_hc_ffn_base_, l_self_modules_layers_modules_14_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_14_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_15_parameters_hc_attn_fn_, l_self_modules_layers_modules_15_parameters_hc_attn_scale_, l_self_modules_layers_modules_15_parameters_hc_attn_base_, l_self_modules_layers_modules_15_modules_attn_norm_parameters_weight_);  getitem_57 = getitem_58 = submod_29 = getitem_59 = l_self_modules_layers_modules_14_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_14_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_14_parameters_hc_ffn_base_ = l_self_modules_layers_modules_14_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_14_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_15_parameters_hc_attn_fn_ = l_self_modules_layers_modules_15_parameters_hc_attn_scale_ = l_self_modules_layers_modules_15_parameters_hc_attn_base_ = l_self_modules_layers_modules_15_modules_attn_norm_parameters_weight_ = None
        getitem_60 = submod_30[0]
        getitem_61 = submod_30[1]
        getitem_62 = submod_30[2]
        getitem_63 = submod_30[3];  submod_30 = None
        submod_31 = self.submod_31(getitem_60, s72, l_positions_, s80);  getitem_60 = None
        submod_32 = self.submod_32(getitem_61, s72, getitem_62, submod_31, getitem_63, l_self_modules_layers_modules_15_parameters_hc_ffn_fn_, l_self_modules_layers_modules_15_parameters_hc_ffn_scale_, l_self_modules_layers_modules_15_parameters_hc_ffn_base_, l_self_modules_layers_modules_15_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_15_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_16_parameters_hc_attn_fn_, l_self_modules_layers_modules_16_parameters_hc_attn_scale_, l_self_modules_layers_modules_16_parameters_hc_attn_base_, l_self_modules_layers_modules_16_modules_attn_norm_parameters_weight_);  getitem_61 = getitem_62 = submod_31 = getitem_63 = l_self_modules_layers_modules_15_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_15_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_15_parameters_hc_ffn_base_ = l_self_modules_layers_modules_15_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_15_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_16_parameters_hc_attn_fn_ = l_self_modules_layers_modules_16_parameters_hc_attn_scale_ = l_self_modules_layers_modules_16_parameters_hc_attn_base_ = l_self_modules_layers_modules_16_modules_attn_norm_parameters_weight_ = None
        getitem_64 = submod_32[0]
        getitem_65 = submod_32[1]
        getitem_66 = submod_32[2]
        getitem_67 = submod_32[3];  submod_32 = None
        submod_33 = self.submod_33(getitem_64, s72, l_positions_, s80);  getitem_64 = None
        submod_34 = self.submod_34(getitem_65, s72, getitem_66, submod_33, getitem_67, l_self_modules_layers_modules_16_parameters_hc_ffn_fn_, l_self_modules_layers_modules_16_parameters_hc_ffn_scale_, l_self_modules_layers_modules_16_parameters_hc_ffn_base_, l_self_modules_layers_modules_16_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_16_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_17_parameters_hc_attn_fn_, l_self_modules_layers_modules_17_parameters_hc_attn_scale_, l_self_modules_layers_modules_17_parameters_hc_attn_base_, l_self_modules_layers_modules_17_modules_attn_norm_parameters_weight_);  getitem_65 = getitem_66 = submod_33 = getitem_67 = l_self_modules_layers_modules_16_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_16_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_16_parameters_hc_ffn_base_ = l_self_modules_layers_modules_16_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_16_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_17_parameters_hc_attn_fn_ = l_self_modules_layers_modules_17_parameters_hc_attn_scale_ = l_self_modules_layers_modules_17_parameters_hc_attn_base_ = l_self_modules_layers_modules_17_modules_attn_norm_parameters_weight_ = None
        getitem_68 = submod_34[0]
        getitem_69 = submod_34[1]
        getitem_70 = submod_34[2]
        getitem_71 = submod_34[3];  submod_34 = None
        submod_35 = self.submod_35(getitem_68, s72, l_positions_, s80);  getitem_68 = None
        submod_36 = self.submod_36(getitem_69, s72, getitem_70, submod_35, getitem_71, l_self_modules_layers_modules_17_parameters_hc_ffn_fn_, l_self_modules_layers_modules_17_parameters_hc_ffn_scale_, l_self_modules_layers_modules_17_parameters_hc_ffn_base_, l_self_modules_layers_modules_17_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_17_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_18_parameters_hc_attn_fn_, l_self_modules_layers_modules_18_parameters_hc_attn_scale_, l_self_modules_layers_modules_18_parameters_hc_attn_base_, l_self_modules_layers_modules_18_modules_attn_norm_parameters_weight_);  getitem_69 = getitem_70 = submod_35 = getitem_71 = l_self_modules_layers_modules_17_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_17_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_17_parameters_hc_ffn_base_ = l_self_modules_layers_modules_17_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_17_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_18_parameters_hc_attn_fn_ = l_self_modules_layers_modules_18_parameters_hc_attn_scale_ = l_self_modules_layers_modules_18_parameters_hc_attn_base_ = l_self_modules_layers_modules_18_modules_attn_norm_parameters_weight_ = None
        getitem_72 = submod_36[0]
        getitem_73 = submod_36[1]
        getitem_74 = submod_36[2]
        getitem_75 = submod_36[3];  submod_36 = None
        submod_37 = self.submod_37(getitem_72, s72, l_positions_, s80);  getitem_72 = None
        submod_38 = self.submod_38(getitem_73, s72, getitem_74, submod_37, getitem_75, l_self_modules_layers_modules_18_parameters_hc_ffn_fn_, l_self_modules_layers_modules_18_parameters_hc_ffn_scale_, l_self_modules_layers_modules_18_parameters_hc_ffn_base_, l_self_modules_layers_modules_18_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_18_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_19_parameters_hc_attn_fn_, l_self_modules_layers_modules_19_parameters_hc_attn_scale_, l_self_modules_layers_modules_19_parameters_hc_attn_base_, l_self_modules_layers_modules_19_modules_attn_norm_parameters_weight_);  getitem_73 = getitem_74 = submod_37 = getitem_75 = l_self_modules_layers_modules_18_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_18_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_18_parameters_hc_ffn_base_ = l_self_modules_layers_modules_18_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_18_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_19_parameters_hc_attn_fn_ = l_self_modules_layers_modules_19_parameters_hc_attn_scale_ = l_self_modules_layers_modules_19_parameters_hc_attn_base_ = l_self_modules_layers_modules_19_modules_attn_norm_parameters_weight_ = None
        getitem_76 = submod_38[0]
        getitem_77 = submod_38[1]
        getitem_78 = submod_38[2]
        getitem_79 = submod_38[3];  submod_38 = None
        submod_39 = self.submod_39(getitem_76, s72, l_positions_, s80);  getitem_76 = None
        submod_40 = self.submod_40(getitem_77, s72, getitem_78, submod_39, getitem_79, l_self_modules_layers_modules_19_parameters_hc_ffn_fn_, l_self_modules_layers_modules_19_parameters_hc_ffn_scale_, l_self_modules_layers_modules_19_parameters_hc_ffn_base_, l_self_modules_layers_modules_19_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_19_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_20_parameters_hc_attn_fn_, l_self_modules_layers_modules_20_parameters_hc_attn_scale_, l_self_modules_layers_modules_20_parameters_hc_attn_base_, l_self_modules_layers_modules_20_modules_attn_norm_parameters_weight_);  getitem_77 = getitem_78 = submod_39 = getitem_79 = l_self_modules_layers_modules_19_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_19_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_19_parameters_hc_ffn_base_ = l_self_modules_layers_modules_19_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_19_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_20_parameters_hc_attn_fn_ = l_self_modules_layers_modules_20_parameters_hc_attn_scale_ = l_self_modules_layers_modules_20_parameters_hc_attn_base_ = l_self_modules_layers_modules_20_modules_attn_norm_parameters_weight_ = None
        getitem_80 = submod_40[0]
        getitem_81 = submod_40[1]
        getitem_82 = submod_40[2]
        getitem_83 = submod_40[3];  submod_40 = None
        submod_41 = self.submod_41(getitem_80, s72, l_positions_, s80);  getitem_80 = None
        submod_42 = self.submod_42(getitem_81, s72, getitem_82, submod_41, getitem_83, l_self_modules_layers_modules_20_parameters_hc_ffn_fn_, l_self_modules_layers_modules_20_parameters_hc_ffn_scale_, l_self_modules_layers_modules_20_parameters_hc_ffn_base_, l_self_modules_layers_modules_20_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_20_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_21_parameters_hc_attn_fn_, l_self_modules_layers_modules_21_parameters_hc_attn_scale_, l_self_modules_layers_modules_21_parameters_hc_attn_base_, l_self_modules_layers_modules_21_modules_attn_norm_parameters_weight_);  getitem_81 = getitem_82 = submod_41 = getitem_83 = l_self_modules_layers_modules_20_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_20_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_20_parameters_hc_ffn_base_ = l_self_modules_layers_modules_20_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_20_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_21_parameters_hc_attn_fn_ = l_self_modules_layers_modules_21_parameters_hc_attn_scale_ = l_self_modules_layers_modules_21_parameters_hc_attn_base_ = l_self_modules_layers_modules_21_modules_attn_norm_parameters_weight_ = None
        getitem_84 = submod_42[0]
        getitem_85 = submod_42[1]
        getitem_86 = submod_42[2]
        getitem_87 = submod_42[3];  submod_42 = None
        submod_43 = self.submod_43(getitem_84, s72, l_positions_, s80);  getitem_84 = None
        submod_44 = self.submod_44(getitem_85, s72, getitem_86, submod_43, getitem_87, l_self_modules_layers_modules_21_parameters_hc_ffn_fn_, l_self_modules_layers_modules_21_parameters_hc_ffn_scale_, l_self_modules_layers_modules_21_parameters_hc_ffn_base_, l_self_modules_layers_modules_21_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_21_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_22_parameters_hc_attn_fn_, l_self_modules_layers_modules_22_parameters_hc_attn_scale_, l_self_modules_layers_modules_22_parameters_hc_attn_base_, l_self_modules_layers_modules_22_modules_attn_norm_parameters_weight_);  getitem_85 = getitem_86 = submod_43 = getitem_87 = l_self_modules_layers_modules_21_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_21_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_21_parameters_hc_ffn_base_ = l_self_modules_layers_modules_21_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_21_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_22_parameters_hc_attn_fn_ = l_self_modules_layers_modules_22_parameters_hc_attn_scale_ = l_self_modules_layers_modules_22_parameters_hc_attn_base_ = l_self_modules_layers_modules_22_modules_attn_norm_parameters_weight_ = None
        getitem_88 = submod_44[0]
        getitem_89 = submod_44[1]
        getitem_90 = submod_44[2]
        getitem_91 = submod_44[3];  submod_44 = None
        submod_45 = self.submod_45(getitem_88, s72, l_positions_, s80);  getitem_88 = None
        submod_46 = self.submod_46(getitem_89, s72, getitem_90, submod_45, getitem_91, l_self_modules_layers_modules_22_parameters_hc_ffn_fn_, l_self_modules_layers_modules_22_parameters_hc_ffn_scale_, l_self_modules_layers_modules_22_parameters_hc_ffn_base_, l_self_modules_layers_modules_22_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_22_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_23_parameters_hc_attn_fn_, l_self_modules_layers_modules_23_parameters_hc_attn_scale_, l_self_modules_layers_modules_23_parameters_hc_attn_base_, l_self_modules_layers_modules_23_modules_attn_norm_parameters_weight_);  getitem_89 = getitem_90 = submod_45 = getitem_91 = l_self_modules_layers_modules_22_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_22_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_22_parameters_hc_ffn_base_ = l_self_modules_layers_modules_22_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_22_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_23_parameters_hc_attn_fn_ = l_self_modules_layers_modules_23_parameters_hc_attn_scale_ = l_self_modules_layers_modules_23_parameters_hc_attn_base_ = l_self_modules_layers_modules_23_modules_attn_norm_parameters_weight_ = None
        getitem_92 = submod_46[0]
        getitem_93 = submod_46[1]
        getitem_94 = submod_46[2]
        getitem_95 = submod_46[3];  submod_46 = None
        submod_47 = self.submod_47(getitem_92, s72, l_positions_, s80);  getitem_92 = None
        submod_48 = self.submod_48(getitem_93, s72, getitem_94, submod_47, getitem_95, l_self_modules_layers_modules_23_parameters_hc_ffn_fn_, l_self_modules_layers_modules_23_parameters_hc_ffn_scale_, l_self_modules_layers_modules_23_parameters_hc_ffn_base_, l_self_modules_layers_modules_23_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_23_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_24_parameters_hc_attn_fn_, l_self_modules_layers_modules_24_parameters_hc_attn_scale_, l_self_modules_layers_modules_24_parameters_hc_attn_base_, l_self_modules_layers_modules_24_modules_attn_norm_parameters_weight_);  getitem_93 = getitem_94 = submod_47 = getitem_95 = l_self_modules_layers_modules_23_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_23_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_23_parameters_hc_ffn_base_ = l_self_modules_layers_modules_23_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_23_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_24_parameters_hc_attn_fn_ = l_self_modules_layers_modules_24_parameters_hc_attn_scale_ = l_self_modules_layers_modules_24_parameters_hc_attn_base_ = l_self_modules_layers_modules_24_modules_attn_norm_parameters_weight_ = None
        getitem_96 = submod_48[0]
        getitem_97 = submod_48[1]
        getitem_98 = submod_48[2]
        getitem_99 = submod_48[3];  submod_48 = None
        submod_49 = self.submod_49(getitem_96, s72, l_positions_, s80);  getitem_96 = None
        submod_50 = self.submod_50(getitem_97, s72, getitem_98, submod_49, getitem_99, l_self_modules_layers_modules_24_parameters_hc_ffn_fn_, l_self_modules_layers_modules_24_parameters_hc_ffn_scale_, l_self_modules_layers_modules_24_parameters_hc_ffn_base_, l_self_modules_layers_modules_24_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_24_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_25_parameters_hc_attn_fn_, l_self_modules_layers_modules_25_parameters_hc_attn_scale_, l_self_modules_layers_modules_25_parameters_hc_attn_base_, l_self_modules_layers_modules_25_modules_attn_norm_parameters_weight_);  getitem_97 = getitem_98 = submod_49 = getitem_99 = l_self_modules_layers_modules_24_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_24_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_24_parameters_hc_ffn_base_ = l_self_modules_layers_modules_24_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_24_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_25_parameters_hc_attn_fn_ = l_self_modules_layers_modules_25_parameters_hc_attn_scale_ = l_self_modules_layers_modules_25_parameters_hc_attn_base_ = l_self_modules_layers_modules_25_modules_attn_norm_parameters_weight_ = None
        getitem_100 = submod_50[0]
        getitem_101 = submod_50[1]
        getitem_102 = submod_50[2]
        getitem_103 = submod_50[3];  submod_50 = None
        submod_51 = self.submod_51(getitem_100, s72, l_positions_, s80);  getitem_100 = None
        submod_52 = self.submod_52(getitem_101, s72, getitem_102, submod_51, getitem_103, l_self_modules_layers_modules_25_parameters_hc_ffn_fn_, l_self_modules_layers_modules_25_parameters_hc_ffn_scale_, l_self_modules_layers_modules_25_parameters_hc_ffn_base_, l_self_modules_layers_modules_25_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_25_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_26_parameters_hc_attn_fn_, l_self_modules_layers_modules_26_parameters_hc_attn_scale_, l_self_modules_layers_modules_26_parameters_hc_attn_base_, l_self_modules_layers_modules_26_modules_attn_norm_parameters_weight_);  getitem_101 = getitem_102 = submod_51 = getitem_103 = l_self_modules_layers_modules_25_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_25_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_25_parameters_hc_ffn_base_ = l_self_modules_layers_modules_25_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_25_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_26_parameters_hc_attn_fn_ = l_self_modules_layers_modules_26_parameters_hc_attn_scale_ = l_self_modules_layers_modules_26_parameters_hc_attn_base_ = l_self_modules_layers_modules_26_modules_attn_norm_parameters_weight_ = None
        getitem_104 = submod_52[0]
        getitem_105 = submod_52[1]
        getitem_106 = submod_52[2]
        getitem_107 = submod_52[3];  submod_52 = None
        submod_53 = self.submod_53(getitem_104, s72, l_positions_, s80);  getitem_104 = None
        submod_54 = self.submod_54(getitem_105, s72, getitem_106, submod_53, getitem_107, l_self_modules_layers_modules_26_parameters_hc_ffn_fn_, l_self_modules_layers_modules_26_parameters_hc_ffn_scale_, l_self_modules_layers_modules_26_parameters_hc_ffn_base_, l_self_modules_layers_modules_26_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_26_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_27_parameters_hc_attn_fn_, l_self_modules_layers_modules_27_parameters_hc_attn_scale_, l_self_modules_layers_modules_27_parameters_hc_attn_base_, l_self_modules_layers_modules_27_modules_attn_norm_parameters_weight_);  getitem_105 = getitem_106 = submod_53 = getitem_107 = l_self_modules_layers_modules_26_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_26_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_26_parameters_hc_ffn_base_ = l_self_modules_layers_modules_26_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_26_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_27_parameters_hc_attn_fn_ = l_self_modules_layers_modules_27_parameters_hc_attn_scale_ = l_self_modules_layers_modules_27_parameters_hc_attn_base_ = l_self_modules_layers_modules_27_modules_attn_norm_parameters_weight_ = None
        getitem_108 = submod_54[0]
        getitem_109 = submod_54[1]
        getitem_110 = submod_54[2]
        getitem_111 = submod_54[3];  submod_54 = None
        submod_55 = self.submod_55(getitem_108, s72, l_positions_, s80);  getitem_108 = None
        submod_56 = self.submod_56(getitem_109, s72, getitem_110, submod_55, getitem_111, l_self_modules_layers_modules_27_parameters_hc_ffn_fn_, l_self_modules_layers_modules_27_parameters_hc_ffn_scale_, l_self_modules_layers_modules_27_parameters_hc_ffn_base_, l_self_modules_layers_modules_27_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_27_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_28_parameters_hc_attn_fn_, l_self_modules_layers_modules_28_parameters_hc_attn_scale_, l_self_modules_layers_modules_28_parameters_hc_attn_base_, l_self_modules_layers_modules_28_modules_attn_norm_parameters_weight_);  getitem_109 = getitem_110 = submod_55 = getitem_111 = l_self_modules_layers_modules_27_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_27_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_27_parameters_hc_ffn_base_ = l_self_modules_layers_modules_27_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_27_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_28_parameters_hc_attn_fn_ = l_self_modules_layers_modules_28_parameters_hc_attn_scale_ = l_self_modules_layers_modules_28_parameters_hc_attn_base_ = l_self_modules_layers_modules_28_modules_attn_norm_parameters_weight_ = None
        getitem_112 = submod_56[0]
        getitem_113 = submod_56[1]
        getitem_114 = submod_56[2]
        getitem_115 = submod_56[3];  submod_56 = None
        submod_57 = self.submod_57(getitem_112, s72, l_positions_, s80);  getitem_112 = None
        submod_58 = self.submod_58(getitem_113, s72, getitem_114, submod_57, getitem_115, l_self_modules_layers_modules_28_parameters_hc_ffn_fn_, l_self_modules_layers_modules_28_parameters_hc_ffn_scale_, l_self_modules_layers_modules_28_parameters_hc_ffn_base_, l_self_modules_layers_modules_28_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_28_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_29_parameters_hc_attn_fn_, l_self_modules_layers_modules_29_parameters_hc_attn_scale_, l_self_modules_layers_modules_29_parameters_hc_attn_base_, l_self_modules_layers_modules_29_modules_attn_norm_parameters_weight_);  getitem_113 = getitem_114 = submod_57 = getitem_115 = l_self_modules_layers_modules_28_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_28_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_28_parameters_hc_ffn_base_ = l_self_modules_layers_modules_28_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_28_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_29_parameters_hc_attn_fn_ = l_self_modules_layers_modules_29_parameters_hc_attn_scale_ = l_self_modules_layers_modules_29_parameters_hc_attn_base_ = l_self_modules_layers_modules_29_modules_attn_norm_parameters_weight_ = None
        getitem_116 = submod_58[0]
        getitem_117 = submod_58[1]
        getitem_118 = submod_58[2]
        getitem_119 = submod_58[3];  submod_58 = None
        submod_59 = self.submod_59(getitem_116, s72, l_positions_, s80);  getitem_116 = None
        submod_60 = self.submod_60(getitem_117, s72, getitem_118, submod_59, getitem_119, l_self_modules_layers_modules_29_parameters_hc_ffn_fn_, l_self_modules_layers_modules_29_parameters_hc_ffn_scale_, l_self_modules_layers_modules_29_parameters_hc_ffn_base_, l_self_modules_layers_modules_29_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_29_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_30_parameters_hc_attn_fn_, l_self_modules_layers_modules_30_parameters_hc_attn_scale_, l_self_modules_layers_modules_30_parameters_hc_attn_base_, l_self_modules_layers_modules_30_modules_attn_norm_parameters_weight_);  getitem_117 = getitem_118 = submod_59 = getitem_119 = l_self_modules_layers_modules_29_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_29_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_29_parameters_hc_ffn_base_ = l_self_modules_layers_modules_29_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_29_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_30_parameters_hc_attn_fn_ = l_self_modules_layers_modules_30_parameters_hc_attn_scale_ = l_self_modules_layers_modules_30_parameters_hc_attn_base_ = l_self_modules_layers_modules_30_modules_attn_norm_parameters_weight_ = None
        getitem_120 = submod_60[0]
        getitem_121 = submod_60[1]
        getitem_122 = submod_60[2]
        getitem_123 = submod_60[3];  submod_60 = None
        submod_61 = self.submod_61(getitem_120, s72, l_positions_, s80);  getitem_120 = None
        submod_62 = self.submod_62(getitem_121, s72, getitem_122, submod_61, getitem_123, l_self_modules_layers_modules_30_parameters_hc_ffn_fn_, l_self_modules_layers_modules_30_parameters_hc_ffn_scale_, l_self_modules_layers_modules_30_parameters_hc_ffn_base_, l_self_modules_layers_modules_30_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_30_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_31_parameters_hc_attn_fn_, l_self_modules_layers_modules_31_parameters_hc_attn_scale_, l_self_modules_layers_modules_31_parameters_hc_attn_base_, l_self_modules_layers_modules_31_modules_attn_norm_parameters_weight_);  getitem_121 = getitem_122 = submod_61 = getitem_123 = l_self_modules_layers_modules_30_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_30_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_30_parameters_hc_ffn_base_ = l_self_modules_layers_modules_30_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_30_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_31_parameters_hc_attn_fn_ = l_self_modules_layers_modules_31_parameters_hc_attn_scale_ = l_self_modules_layers_modules_31_parameters_hc_attn_base_ = l_self_modules_layers_modules_31_modules_attn_norm_parameters_weight_ = None
        getitem_124 = submod_62[0]
        getitem_125 = submod_62[1]
        getitem_126 = submod_62[2]
        getitem_127 = submod_62[3];  submod_62 = None
        submod_63 = self.submod_63(getitem_124, s72, l_positions_, s80);  getitem_124 = None
        submod_64 = self.submod_64(getitem_125, s72, getitem_126, submod_63, getitem_127, l_self_modules_layers_modules_31_parameters_hc_ffn_fn_, l_self_modules_layers_modules_31_parameters_hc_ffn_scale_, l_self_modules_layers_modules_31_parameters_hc_ffn_base_, l_self_modules_layers_modules_31_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_31_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_32_parameters_hc_attn_fn_, l_self_modules_layers_modules_32_parameters_hc_attn_scale_, l_self_modules_layers_modules_32_parameters_hc_attn_base_, l_self_modules_layers_modules_32_modules_attn_norm_parameters_weight_);  getitem_125 = getitem_126 = submod_63 = getitem_127 = l_self_modules_layers_modules_31_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_31_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_31_parameters_hc_ffn_base_ = l_self_modules_layers_modules_31_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_31_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_32_parameters_hc_attn_fn_ = l_self_modules_layers_modules_32_parameters_hc_attn_scale_ = l_self_modules_layers_modules_32_parameters_hc_attn_base_ = l_self_modules_layers_modules_32_modules_attn_norm_parameters_weight_ = None
        getitem_128 = submod_64[0]
        getitem_129 = submod_64[1]
        getitem_130 = submod_64[2]
        getitem_131 = submod_64[3];  submod_64 = None
        submod_65 = self.submod_65(getitem_128, s72, l_positions_, s80);  getitem_128 = None
        submod_66 = self.submod_66(getitem_129, s72, getitem_130, submod_65, getitem_131, l_self_modules_layers_modules_32_parameters_hc_ffn_fn_, l_self_modules_layers_modules_32_parameters_hc_ffn_scale_, l_self_modules_layers_modules_32_parameters_hc_ffn_base_, l_self_modules_layers_modules_32_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_32_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_33_parameters_hc_attn_fn_, l_self_modules_layers_modules_33_parameters_hc_attn_scale_, l_self_modules_layers_modules_33_parameters_hc_attn_base_, l_self_modules_layers_modules_33_modules_attn_norm_parameters_weight_);  getitem_129 = getitem_130 = submod_65 = getitem_131 = l_self_modules_layers_modules_32_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_32_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_32_parameters_hc_ffn_base_ = l_self_modules_layers_modules_32_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_32_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_33_parameters_hc_attn_fn_ = l_self_modules_layers_modules_33_parameters_hc_attn_scale_ = l_self_modules_layers_modules_33_parameters_hc_attn_base_ = l_self_modules_layers_modules_33_modules_attn_norm_parameters_weight_ = None
        getitem_132 = submod_66[0]
        getitem_133 = submod_66[1]
        getitem_134 = submod_66[2]
        getitem_135 = submod_66[3];  submod_66 = None
        submod_67 = self.submod_67(getitem_132, s72, l_positions_, s80);  getitem_132 = None
        submod_68 = self.submod_68(getitem_133, s72, getitem_134, submod_67, getitem_135, l_self_modules_layers_modules_33_parameters_hc_ffn_fn_, l_self_modules_layers_modules_33_parameters_hc_ffn_scale_, l_self_modules_layers_modules_33_parameters_hc_ffn_base_, l_self_modules_layers_modules_33_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_33_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_34_parameters_hc_attn_fn_, l_self_modules_layers_modules_34_parameters_hc_attn_scale_, l_self_modules_layers_modules_34_parameters_hc_attn_base_, l_self_modules_layers_modules_34_modules_attn_norm_parameters_weight_);  getitem_133 = getitem_134 = submod_67 = getitem_135 = l_self_modules_layers_modules_33_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_33_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_33_parameters_hc_ffn_base_ = l_self_modules_layers_modules_33_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_33_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_34_parameters_hc_attn_fn_ = l_self_modules_layers_modules_34_parameters_hc_attn_scale_ = l_self_modules_layers_modules_34_parameters_hc_attn_base_ = l_self_modules_layers_modules_34_modules_attn_norm_parameters_weight_ = None
        getitem_136 = submod_68[0]
        getitem_137 = submod_68[1]
        getitem_138 = submod_68[2]
        getitem_139 = submod_68[3];  submod_68 = None
        submod_69 = self.submod_69(getitem_136, s72, l_positions_, s80);  getitem_136 = None
        submod_70 = self.submod_70(getitem_137, s72, getitem_138, submod_69, getitem_139, l_self_modules_layers_modules_34_parameters_hc_ffn_fn_, l_self_modules_layers_modules_34_parameters_hc_ffn_scale_, l_self_modules_layers_modules_34_parameters_hc_ffn_base_, l_self_modules_layers_modules_34_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_34_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_35_parameters_hc_attn_fn_, l_self_modules_layers_modules_35_parameters_hc_attn_scale_, l_self_modules_layers_modules_35_parameters_hc_attn_base_, l_self_modules_layers_modules_35_modules_attn_norm_parameters_weight_);  getitem_137 = getitem_138 = submod_69 = getitem_139 = l_self_modules_layers_modules_34_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_34_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_34_parameters_hc_ffn_base_ = l_self_modules_layers_modules_34_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_34_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_35_parameters_hc_attn_fn_ = l_self_modules_layers_modules_35_parameters_hc_attn_scale_ = l_self_modules_layers_modules_35_parameters_hc_attn_base_ = l_self_modules_layers_modules_35_modules_attn_norm_parameters_weight_ = None
        getitem_140 = submod_70[0]
        getitem_141 = submod_70[1]
        getitem_142 = submod_70[2]
        getitem_143 = submod_70[3];  submod_70 = None
        submod_71 = self.submod_71(getitem_140, s72, l_positions_, s80);  getitem_140 = None
        submod_72 = self.submod_72(getitem_141, s72, getitem_142, submod_71, getitem_143, l_self_modules_layers_modules_35_parameters_hc_ffn_fn_, l_self_modules_layers_modules_35_parameters_hc_ffn_scale_, l_self_modules_layers_modules_35_parameters_hc_ffn_base_, l_self_modules_layers_modules_35_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_35_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_36_parameters_hc_attn_fn_, l_self_modules_layers_modules_36_parameters_hc_attn_scale_, l_self_modules_layers_modules_36_parameters_hc_attn_base_, l_self_modules_layers_modules_36_modules_attn_norm_parameters_weight_);  getitem_141 = getitem_142 = submod_71 = getitem_143 = l_self_modules_layers_modules_35_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_35_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_35_parameters_hc_ffn_base_ = l_self_modules_layers_modules_35_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_35_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_36_parameters_hc_attn_fn_ = l_self_modules_layers_modules_36_parameters_hc_attn_scale_ = l_self_modules_layers_modules_36_parameters_hc_attn_base_ = l_self_modules_layers_modules_36_modules_attn_norm_parameters_weight_ = None
        getitem_144 = submod_72[0]
        getitem_145 = submod_72[1]
        getitem_146 = submod_72[2]
        getitem_147 = submod_72[3];  submod_72 = None
        submod_73 = self.submod_73(getitem_144, s72, l_positions_, s80);  getitem_144 = None
        submod_74 = self.submod_74(getitem_145, s72, getitem_146, submod_73, getitem_147, l_self_modules_layers_modules_36_parameters_hc_ffn_fn_, l_self_modules_layers_modules_36_parameters_hc_ffn_scale_, l_self_modules_layers_modules_36_parameters_hc_ffn_base_, l_self_modules_layers_modules_36_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_36_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_37_parameters_hc_attn_fn_, l_self_modules_layers_modules_37_parameters_hc_attn_scale_, l_self_modules_layers_modules_37_parameters_hc_attn_base_, l_self_modules_layers_modules_37_modules_attn_norm_parameters_weight_);  getitem_145 = getitem_146 = submod_73 = getitem_147 = l_self_modules_layers_modules_36_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_36_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_36_parameters_hc_ffn_base_ = l_self_modules_layers_modules_36_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_36_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_37_parameters_hc_attn_fn_ = l_self_modules_layers_modules_37_parameters_hc_attn_scale_ = l_self_modules_layers_modules_37_parameters_hc_attn_base_ = l_self_modules_layers_modules_37_modules_attn_norm_parameters_weight_ = None
        getitem_148 = submod_74[0]
        getitem_149 = submod_74[1]
        getitem_150 = submod_74[2]
        getitem_151 = submod_74[3];  submod_74 = None
        submod_75 = self.submod_75(getitem_148, s72, l_positions_, s80);  getitem_148 = None
        submod_76 = self.submod_76(getitem_149, s72, getitem_150, submod_75, getitem_151, l_self_modules_layers_modules_37_parameters_hc_ffn_fn_, l_self_modules_layers_modules_37_parameters_hc_ffn_scale_, l_self_modules_layers_modules_37_parameters_hc_ffn_base_, l_self_modules_layers_modules_37_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_37_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_38_parameters_hc_attn_fn_, l_self_modules_layers_modules_38_parameters_hc_attn_scale_, l_self_modules_layers_modules_38_parameters_hc_attn_base_, l_self_modules_layers_modules_38_modules_attn_norm_parameters_weight_);  getitem_149 = getitem_150 = submod_75 = getitem_151 = l_self_modules_layers_modules_37_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_37_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_37_parameters_hc_ffn_base_ = l_self_modules_layers_modules_37_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_37_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_38_parameters_hc_attn_fn_ = l_self_modules_layers_modules_38_parameters_hc_attn_scale_ = l_self_modules_layers_modules_38_parameters_hc_attn_base_ = l_self_modules_layers_modules_38_modules_attn_norm_parameters_weight_ = None
        getitem_152 = submod_76[0]
        getitem_153 = submod_76[1]
        getitem_154 = submod_76[2]
        getitem_155 = submod_76[3];  submod_76 = None
        submod_77 = self.submod_77(getitem_152, s72, l_positions_, s80);  getitem_152 = None
        submod_78 = self.submod_78(getitem_153, s72, getitem_154, submod_77, getitem_155, l_self_modules_layers_modules_38_parameters_hc_ffn_fn_, l_self_modules_layers_modules_38_parameters_hc_ffn_scale_, l_self_modules_layers_modules_38_parameters_hc_ffn_base_, l_self_modules_layers_modules_38_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_38_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_39_parameters_hc_attn_fn_, l_self_modules_layers_modules_39_parameters_hc_attn_scale_, l_self_modules_layers_modules_39_parameters_hc_attn_base_, l_self_modules_layers_modules_39_modules_attn_norm_parameters_weight_);  getitem_153 = getitem_154 = submod_77 = getitem_155 = l_self_modules_layers_modules_38_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_38_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_38_parameters_hc_ffn_base_ = l_self_modules_layers_modules_38_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_38_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_39_parameters_hc_attn_fn_ = l_self_modules_layers_modules_39_parameters_hc_attn_scale_ = l_self_modules_layers_modules_39_parameters_hc_attn_base_ = l_self_modules_layers_modules_39_modules_attn_norm_parameters_weight_ = None
        getitem_156 = submod_78[0]
        getitem_157 = submod_78[1]
        getitem_158 = submod_78[2]
        getitem_159 = submod_78[3];  submod_78 = None
        submod_79 = self.submod_79(getitem_156, s72, l_positions_, s80);  getitem_156 = None
        submod_80 = self.submod_80(getitem_157, s72, getitem_158, submod_79, getitem_159, l_self_modules_layers_modules_39_parameters_hc_ffn_fn_, l_self_modules_layers_modules_39_parameters_hc_ffn_scale_, l_self_modules_layers_modules_39_parameters_hc_ffn_base_, l_self_modules_layers_modules_39_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_39_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_40_parameters_hc_attn_fn_, l_self_modules_layers_modules_40_parameters_hc_attn_scale_, l_self_modules_layers_modules_40_parameters_hc_attn_base_, l_self_modules_layers_modules_40_modules_attn_norm_parameters_weight_);  getitem_157 = getitem_158 = submod_79 = getitem_159 = l_self_modules_layers_modules_39_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_39_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_39_parameters_hc_ffn_base_ = l_self_modules_layers_modules_39_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_39_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_40_parameters_hc_attn_fn_ = l_self_modules_layers_modules_40_parameters_hc_attn_scale_ = l_self_modules_layers_modules_40_parameters_hc_attn_base_ = l_self_modules_layers_modules_40_modules_attn_norm_parameters_weight_ = None
        getitem_160 = submod_80[0]
        getitem_161 = submod_80[1]
        getitem_162 = submod_80[2]
        getitem_163 = submod_80[3];  submod_80 = None
        submod_81 = self.submod_81(getitem_160, s72, l_positions_, s80);  getitem_160 = None
        submod_82 = self.submod_82(getitem_161, s72, getitem_162, submod_81, getitem_163, l_self_modules_layers_modules_40_parameters_hc_ffn_fn_, l_self_modules_layers_modules_40_parameters_hc_ffn_scale_, l_self_modules_layers_modules_40_parameters_hc_ffn_base_, l_self_modules_layers_modules_40_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_40_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_41_parameters_hc_attn_fn_, l_self_modules_layers_modules_41_parameters_hc_attn_scale_, l_self_modules_layers_modules_41_parameters_hc_attn_base_, l_self_modules_layers_modules_41_modules_attn_norm_parameters_weight_);  getitem_161 = getitem_162 = submod_81 = getitem_163 = l_self_modules_layers_modules_40_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_40_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_40_parameters_hc_ffn_base_ = l_self_modules_layers_modules_40_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_40_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_41_parameters_hc_attn_fn_ = l_self_modules_layers_modules_41_parameters_hc_attn_scale_ = l_self_modules_layers_modules_41_parameters_hc_attn_base_ = l_self_modules_layers_modules_41_modules_attn_norm_parameters_weight_ = None
        getitem_164 = submod_82[0]
        getitem_165 = submod_82[1]
        getitem_166 = submod_82[2]
        getitem_167 = submod_82[3];  submod_82 = None
        submod_83 = self.submod_83(getitem_164, s72, l_positions_, s80);  getitem_164 = None
        submod_84 = self.submod_84(getitem_165, s72, getitem_166, submod_83, getitem_167, l_self_modules_layers_modules_41_parameters_hc_ffn_fn_, l_self_modules_layers_modules_41_parameters_hc_ffn_scale_, l_self_modules_layers_modules_41_parameters_hc_ffn_base_, l_self_modules_layers_modules_41_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_41_modules_ffn_modules_gate_parameters_weight_, l_self_modules_layers_modules_42_parameters_hc_attn_fn_, l_self_modules_layers_modules_42_parameters_hc_attn_scale_, l_self_modules_layers_modules_42_parameters_hc_attn_base_, l_self_modules_layers_modules_42_modules_attn_norm_parameters_weight_);  getitem_165 = getitem_166 = submod_83 = getitem_167 = l_self_modules_layers_modules_41_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_41_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_41_parameters_hc_ffn_base_ = l_self_modules_layers_modules_41_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_41_modules_ffn_modules_gate_parameters_weight_ = l_self_modules_layers_modules_42_parameters_hc_attn_fn_ = l_self_modules_layers_modules_42_parameters_hc_attn_scale_ = l_self_modules_layers_modules_42_parameters_hc_attn_base_ = l_self_modules_layers_modules_42_modules_attn_norm_parameters_weight_ = None
        getitem_168 = submod_84[0]
        getitem_169 = submod_84[1]
        getitem_170 = submod_84[2]
        getitem_171 = submod_84[3];  submod_84 = None
        submod_85 = self.submod_85(getitem_168, s72, l_positions_, s80);  getitem_168 = l_positions_ = s80 = None
        submod_86 = self.submod_86(getitem_169, s72, getitem_170, submod_85, getitem_171, l_self_modules_layers_modules_42_parameters_hc_ffn_fn_, l_self_modules_layers_modules_42_parameters_hc_ffn_scale_, l_self_modules_layers_modules_42_parameters_hc_ffn_base_, l_self_modules_layers_modules_42_modules_ffn_norm_parameters_weight_, l_self_modules_layers_modules_42_modules_ffn_modules_gate_parameters_weight_);  getitem_169 = s72 = getitem_170 = submod_85 = getitem_171 = l_self_modules_layers_modules_42_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_42_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_42_parameters_hc_ffn_base_ = l_self_modules_layers_modules_42_modules_ffn_norm_parameters_weight_ = l_self_modules_layers_modules_42_modules_ffn_modules_gate_parameters_weight_ = None
        return (submod_86,)
        
    class submod_0(torch.nn.Module):
        def forward(self, l_input_ids_: "i32[s72]", s72: "Sym(s72)", l_self_modules_embed_parameters_weight_: "bf16[32320, 4096]", l_self_modules_layers_modules_0_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_0_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_0_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_0_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            masked_embedding: "bf16[s72, 4096]" = torch.ops.aiter.masked_embedding(l_input_ids_, l_self_modules_embed_parameters_weight_, 96960, 129280);  l_input_ids_ = l_self_modules_embed_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(masked_embedding, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  masked_embedding = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "bf16[s72, 1, 4096]" = all_reduce_.unsqueeze(-2);  all_reduce_ = None
            repeat: "bf16[s72, 4, 4096]" = unsqueeze.repeat(1, 4, 1);  unsqueeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(repeat, l_self_modules_layers_modules_0_parameters_hc_attn_fn_, l_self_modules_layers_modules_0_parameters_hc_attn_scale_, l_self_modules_layers_modules_0_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_0_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_0_parameters_hc_attn_fn_ = l_self_modules_layers_modules_0_parameters_hc_attn_scale_ = l_self_modules_layers_modules_0_parameters_hc_attn_base_ = l_self_modules_layers_modules_0_modules_attn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            return (getitem_2, repeat, squeeze, getitem_1)
            
    class submod_1(torch.nn.Module):
        def forward(self, y: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y, l_positions_, 'layers.0.attn');  y = l_positions_ = None
            return v4_attention_with_output
            
    class submod_2(torch.nn.Module):
        def forward(self, h: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_1: "f32[s72, 4]", x: "bf16[s72, 4096]", comb: "f32[s72, 4, 4]", l_self_modules_layers_modules_0_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_0_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_0_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_0_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_0_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_1_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_1_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_1_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_1_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(h)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_1.unsqueeze(-1);  post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x, h, unsqueeze, comb);  x = h = unsqueeze = comb = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_0_parameters_hc_ffn_fn_, l_self_modules_layers_modules_0_parameters_hc_ffn_scale_, l_self_modules_layers_modules_0_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_0_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_0_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_0_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_0_parameters_hc_ffn_base_ = l_self_modules_layers_modules_0_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_0_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_0_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.0.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_1_parameters_hc_attn_fn_, l_self_modules_layers_modules_1_parameters_hc_attn_scale_, l_self_modules_layers_modules_1_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_1_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_1_parameters_hc_attn_fn_ = l_self_modules_layers_modules_1_parameters_hc_attn_scale_ = l_self_modules_layers_modules_1_parameters_hc_attn_base_ = l_self_modules_layers_modules_1_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_3(torch.nn.Module):
        def forward(self, y_2: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_2, l_positions_, 'layers.1.attn');  y_2 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_4(torch.nn.Module):
        def forward(self, out_1: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_5: "f32[s72, 4]", x_1: "bf16[s72, 4096]", comb_2: "f32[s72, 4, 4]", l_self_modules_layers_modules_1_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_1_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_1_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_1_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_1_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_2_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_2_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_2_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_2_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_1)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_5.unsqueeze(-1);  post_5 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_1, out_1, unsqueeze, comb_2);  x_1 = out_1 = unsqueeze = comb_2 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_1_parameters_hc_ffn_fn_, l_self_modules_layers_modules_1_parameters_hc_ffn_scale_, l_self_modules_layers_modules_1_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_1_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_1_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_1_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_1_parameters_hc_ffn_base_ = l_self_modules_layers_modules_1_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_1_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_1_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.1.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_2_parameters_hc_attn_fn_, l_self_modules_layers_modules_2_parameters_hc_attn_scale_, l_self_modules_layers_modules_2_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_2_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_2_parameters_hc_attn_fn_ = l_self_modules_layers_modules_2_parameters_hc_attn_scale_ = l_self_modules_layers_modules_2_parameters_hc_attn_base_ = l_self_modules_layers_modules_2_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_5(torch.nn.Module):
        def forward(self, y_4: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_4, l_positions_, 'layers.2.attn');  y_4 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_6(torch.nn.Module):
        def forward(self, out_3: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_9: "f32[s72, 4]", x_2: "bf16[s72, 4096]", comb_4: "f32[s72, 4, 4]", l_self_modules_layers_modules_2_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_2_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_2_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_2_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_2_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_3_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_3_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_3_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_3_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_3)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_9.unsqueeze(-1);  post_9 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_2, out_3, unsqueeze, comb_4);  x_2 = out_3 = unsqueeze = comb_4 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_2_parameters_hc_ffn_fn_, l_self_modules_layers_modules_2_parameters_hc_ffn_scale_, l_self_modules_layers_modules_2_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_2_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_2_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_2_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_2_parameters_hc_ffn_base_ = l_self_modules_layers_modules_2_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_2_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_2_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.2.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_3_parameters_hc_attn_fn_, l_self_modules_layers_modules_3_parameters_hc_attn_scale_, l_self_modules_layers_modules_3_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_3_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_3_parameters_hc_attn_fn_ = l_self_modules_layers_modules_3_parameters_hc_attn_scale_ = l_self_modules_layers_modules_3_parameters_hc_attn_base_ = l_self_modules_layers_modules_3_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_7(torch.nn.Module):
        def forward(self, y_6: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_6, l_positions_, 'layers.3.attn');  y_6 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_8(torch.nn.Module):
        def forward(self, out_5: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_13: "f32[s72, 4]", x_3: "bf16[s72, 4096]", comb_6: "f32[s72, 4, 4]", l_self_modules_layers_modules_3_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_3_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_3_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_3_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_3_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_4_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_4_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_4_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_4_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_5)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_13.unsqueeze(-1);  post_13 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_3, out_5, unsqueeze, comb_6);  x_3 = out_5 = unsqueeze = comb_6 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_3_parameters_hc_ffn_fn_, l_self_modules_layers_modules_3_parameters_hc_ffn_scale_, l_self_modules_layers_modules_3_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_3_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_3_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_3_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_3_parameters_hc_ffn_base_ = l_self_modules_layers_modules_3_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_3_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_3_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.3.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_4_parameters_hc_attn_fn_, l_self_modules_layers_modules_4_parameters_hc_attn_scale_, l_self_modules_layers_modules_4_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_4_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_4_parameters_hc_attn_fn_ = l_self_modules_layers_modules_4_parameters_hc_attn_scale_ = l_self_modules_layers_modules_4_parameters_hc_attn_base_ = l_self_modules_layers_modules_4_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_9(torch.nn.Module):
        def forward(self, y_8: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_8, l_positions_, 'layers.4.attn');  y_8 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_10(torch.nn.Module):
        def forward(self, out_7: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_17: "f32[s72, 4]", x_4: "bf16[s72, 4096]", comb_8: "f32[s72, 4, 4]", l_self_modules_layers_modules_4_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_4_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_4_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_4_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_4_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_5_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_5_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_5_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_5_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_7)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_17.unsqueeze(-1);  post_17 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_4, out_7, unsqueeze, comb_8);  x_4 = out_7 = unsqueeze = comb_8 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_4_parameters_hc_ffn_fn_, l_self_modules_layers_modules_4_parameters_hc_ffn_scale_, l_self_modules_layers_modules_4_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_4_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_4_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_4_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_4_parameters_hc_ffn_base_ = l_self_modules_layers_modules_4_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_4_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_4_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.4.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_5_parameters_hc_attn_fn_, l_self_modules_layers_modules_5_parameters_hc_attn_scale_, l_self_modules_layers_modules_5_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_5_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_5_parameters_hc_attn_fn_ = l_self_modules_layers_modules_5_parameters_hc_attn_scale_ = l_self_modules_layers_modules_5_parameters_hc_attn_base_ = l_self_modules_layers_modules_5_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_11(torch.nn.Module):
        def forward(self, y_10: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_10, l_positions_, 'layers.5.attn');  y_10 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_12(torch.nn.Module):
        def forward(self, out_9: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_21: "f32[s72, 4]", x_5: "bf16[s72, 4096]", comb_10: "f32[s72, 4, 4]", l_self_modules_layers_modules_5_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_5_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_5_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_5_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_5_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_6_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_6_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_6_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_6_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_9)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_21.unsqueeze(-1);  post_21 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_5, out_9, unsqueeze, comb_10);  x_5 = out_9 = unsqueeze = comb_10 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_5_parameters_hc_ffn_fn_, l_self_modules_layers_modules_5_parameters_hc_ffn_scale_, l_self_modules_layers_modules_5_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_5_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_5_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_5_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_5_parameters_hc_ffn_base_ = l_self_modules_layers_modules_5_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_5_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_5_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.5.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_6_parameters_hc_attn_fn_, l_self_modules_layers_modules_6_parameters_hc_attn_scale_, l_self_modules_layers_modules_6_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_6_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_6_parameters_hc_attn_fn_ = l_self_modules_layers_modules_6_parameters_hc_attn_scale_ = l_self_modules_layers_modules_6_parameters_hc_attn_base_ = l_self_modules_layers_modules_6_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_13(torch.nn.Module):
        def forward(self, y_12: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_12, l_positions_, 'layers.6.attn');  y_12 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_14(torch.nn.Module):
        def forward(self, out_11: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_25: "f32[s72, 4]", x_6: "bf16[s72, 4096]", comb_12: "f32[s72, 4, 4]", l_self_modules_layers_modules_6_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_6_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_6_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_6_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_6_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_7_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_7_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_7_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_7_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_11)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_25.unsqueeze(-1);  post_25 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_6, out_11, unsqueeze, comb_12);  x_6 = out_11 = unsqueeze = comb_12 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_6_parameters_hc_ffn_fn_, l_self_modules_layers_modules_6_parameters_hc_ffn_scale_, l_self_modules_layers_modules_6_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_6_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_6_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_6_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_6_parameters_hc_ffn_base_ = l_self_modules_layers_modules_6_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_6_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_6_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.6.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_7_parameters_hc_attn_fn_, l_self_modules_layers_modules_7_parameters_hc_attn_scale_, l_self_modules_layers_modules_7_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_7_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_7_parameters_hc_attn_fn_ = l_self_modules_layers_modules_7_parameters_hc_attn_scale_ = l_self_modules_layers_modules_7_parameters_hc_attn_base_ = l_self_modules_layers_modules_7_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_15(torch.nn.Module):
        def forward(self, y_14: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_14, l_positions_, 'layers.7.attn');  y_14 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_16(torch.nn.Module):
        def forward(self, out_13: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_29: "f32[s72, 4]", x_7: "bf16[s72, 4096]", comb_14: "f32[s72, 4, 4]", l_self_modules_layers_modules_7_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_7_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_7_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_7_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_7_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_8_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_8_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_8_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_8_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_13)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_29.unsqueeze(-1);  post_29 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_7, out_13, unsqueeze, comb_14);  x_7 = out_13 = unsqueeze = comb_14 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_7_parameters_hc_ffn_fn_, l_self_modules_layers_modules_7_parameters_hc_ffn_scale_, l_self_modules_layers_modules_7_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_7_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_7_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_7_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_7_parameters_hc_ffn_base_ = l_self_modules_layers_modules_7_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_7_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_7_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.7.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_8_parameters_hc_attn_fn_, l_self_modules_layers_modules_8_parameters_hc_attn_scale_, l_self_modules_layers_modules_8_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_8_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_8_parameters_hc_attn_fn_ = l_self_modules_layers_modules_8_parameters_hc_attn_scale_ = l_self_modules_layers_modules_8_parameters_hc_attn_base_ = l_self_modules_layers_modules_8_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_17(torch.nn.Module):
        def forward(self, y_16: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_16, l_positions_, 'layers.8.attn');  y_16 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_18(torch.nn.Module):
        def forward(self, out_15: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_33: "f32[s72, 4]", x_8: "bf16[s72, 4096]", comb_16: "f32[s72, 4, 4]", l_self_modules_layers_modules_8_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_8_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_8_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_8_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_8_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_9_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_9_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_9_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_9_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_15)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_33.unsqueeze(-1);  post_33 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_8, out_15, unsqueeze, comb_16);  x_8 = out_15 = unsqueeze = comb_16 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_8_parameters_hc_ffn_fn_, l_self_modules_layers_modules_8_parameters_hc_ffn_scale_, l_self_modules_layers_modules_8_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_8_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_8_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_8_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_8_parameters_hc_ffn_base_ = l_self_modules_layers_modules_8_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_8_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_8_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.8.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_9_parameters_hc_attn_fn_, l_self_modules_layers_modules_9_parameters_hc_attn_scale_, l_self_modules_layers_modules_9_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_9_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_9_parameters_hc_attn_fn_ = l_self_modules_layers_modules_9_parameters_hc_attn_scale_ = l_self_modules_layers_modules_9_parameters_hc_attn_base_ = l_self_modules_layers_modules_9_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_19(torch.nn.Module):
        def forward(self, y_18: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_18, l_positions_, 'layers.9.attn');  y_18 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_20(torch.nn.Module):
        def forward(self, out_17: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_37: "f32[s72, 4]", x_9: "bf16[s72, 4096]", comb_18: "f32[s72, 4, 4]", l_self_modules_layers_modules_9_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_9_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_9_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_9_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_9_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_10_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_10_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_10_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_10_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_17)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_37.unsqueeze(-1);  post_37 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_9, out_17, unsqueeze, comb_18);  x_9 = out_17 = unsqueeze = comb_18 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_9_parameters_hc_ffn_fn_, l_self_modules_layers_modules_9_parameters_hc_ffn_scale_, l_self_modules_layers_modules_9_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_9_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_9_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_9_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_9_parameters_hc_ffn_base_ = l_self_modules_layers_modules_9_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_9_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_9_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.9.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_10_parameters_hc_attn_fn_, l_self_modules_layers_modules_10_parameters_hc_attn_scale_, l_self_modules_layers_modules_10_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_10_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_10_parameters_hc_attn_fn_ = l_self_modules_layers_modules_10_parameters_hc_attn_scale_ = l_self_modules_layers_modules_10_parameters_hc_attn_base_ = l_self_modules_layers_modules_10_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_21(torch.nn.Module):
        def forward(self, y_20: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_20, l_positions_, 'layers.10.attn');  y_20 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_22(torch.nn.Module):
        def forward(self, out_19: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_41: "f32[s72, 4]", x_10: "bf16[s72, 4096]", comb_20: "f32[s72, 4, 4]", l_self_modules_layers_modules_10_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_10_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_10_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_10_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_10_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_11_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_11_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_11_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_11_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_19)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_41.unsqueeze(-1);  post_41 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_10, out_19, unsqueeze, comb_20);  x_10 = out_19 = unsqueeze = comb_20 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_10_parameters_hc_ffn_fn_, l_self_modules_layers_modules_10_parameters_hc_ffn_scale_, l_self_modules_layers_modules_10_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_10_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_10_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_10_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_10_parameters_hc_ffn_base_ = l_self_modules_layers_modules_10_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_10_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_10_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.10.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_11_parameters_hc_attn_fn_, l_self_modules_layers_modules_11_parameters_hc_attn_scale_, l_self_modules_layers_modules_11_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_11_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_11_parameters_hc_attn_fn_ = l_self_modules_layers_modules_11_parameters_hc_attn_scale_ = l_self_modules_layers_modules_11_parameters_hc_attn_base_ = l_self_modules_layers_modules_11_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_23(torch.nn.Module):
        def forward(self, y_22: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_22, l_positions_, 'layers.11.attn');  y_22 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_24(torch.nn.Module):
        def forward(self, out_21: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_45: "f32[s72, 4]", x_11: "bf16[s72, 4096]", comb_22: "f32[s72, 4, 4]", l_self_modules_layers_modules_11_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_11_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_11_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_11_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_11_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_12_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_12_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_12_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_12_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_21)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_45.unsqueeze(-1);  post_45 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_11, out_21, unsqueeze, comb_22);  x_11 = out_21 = unsqueeze = comb_22 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_11_parameters_hc_ffn_fn_, l_self_modules_layers_modules_11_parameters_hc_ffn_scale_, l_self_modules_layers_modules_11_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_11_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_11_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_11_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_11_parameters_hc_ffn_base_ = l_self_modules_layers_modules_11_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_11_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_11_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.11.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_12_parameters_hc_attn_fn_, l_self_modules_layers_modules_12_parameters_hc_attn_scale_, l_self_modules_layers_modules_12_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_12_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_12_parameters_hc_attn_fn_ = l_self_modules_layers_modules_12_parameters_hc_attn_scale_ = l_self_modules_layers_modules_12_parameters_hc_attn_base_ = l_self_modules_layers_modules_12_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_25(torch.nn.Module):
        def forward(self, y_24: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_24, l_positions_, 'layers.12.attn');  y_24 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_26(torch.nn.Module):
        def forward(self, out_23: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_49: "f32[s72, 4]", x_12: "bf16[s72, 4096]", comb_24: "f32[s72, 4, 4]", l_self_modules_layers_modules_12_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_12_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_12_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_12_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_12_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_13_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_13_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_13_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_13_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_23)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_49.unsqueeze(-1);  post_49 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_12, out_23, unsqueeze, comb_24);  x_12 = out_23 = unsqueeze = comb_24 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_12_parameters_hc_ffn_fn_, l_self_modules_layers_modules_12_parameters_hc_ffn_scale_, l_self_modules_layers_modules_12_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_12_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_12_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_12_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_12_parameters_hc_ffn_base_ = l_self_modules_layers_modules_12_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_12_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_12_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.12.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_13_parameters_hc_attn_fn_, l_self_modules_layers_modules_13_parameters_hc_attn_scale_, l_self_modules_layers_modules_13_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_13_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_13_parameters_hc_attn_fn_ = l_self_modules_layers_modules_13_parameters_hc_attn_scale_ = l_self_modules_layers_modules_13_parameters_hc_attn_base_ = l_self_modules_layers_modules_13_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_27(torch.nn.Module):
        def forward(self, y_26: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_26, l_positions_, 'layers.13.attn');  y_26 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_28(torch.nn.Module):
        def forward(self, out_25: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_53: "f32[s72, 4]", x_13: "bf16[s72, 4096]", comb_26: "f32[s72, 4, 4]", l_self_modules_layers_modules_13_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_13_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_13_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_13_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_13_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_14_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_14_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_14_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_14_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_25)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_53.unsqueeze(-1);  post_53 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_13, out_25, unsqueeze, comb_26);  x_13 = out_25 = unsqueeze = comb_26 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_13_parameters_hc_ffn_fn_, l_self_modules_layers_modules_13_parameters_hc_ffn_scale_, l_self_modules_layers_modules_13_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_13_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_13_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_13_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_13_parameters_hc_ffn_base_ = l_self_modules_layers_modules_13_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_13_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_13_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.13.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_14_parameters_hc_attn_fn_, l_self_modules_layers_modules_14_parameters_hc_attn_scale_, l_self_modules_layers_modules_14_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_14_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_14_parameters_hc_attn_fn_ = l_self_modules_layers_modules_14_parameters_hc_attn_scale_ = l_self_modules_layers_modules_14_parameters_hc_attn_base_ = l_self_modules_layers_modules_14_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_29(torch.nn.Module):
        def forward(self, y_28: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_28, l_positions_, 'layers.14.attn');  y_28 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_30(torch.nn.Module):
        def forward(self, out_27: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_57: "f32[s72, 4]", x_14: "bf16[s72, 4096]", comb_28: "f32[s72, 4, 4]", l_self_modules_layers_modules_14_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_14_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_14_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_14_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_14_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_15_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_15_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_15_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_15_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_27)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_57.unsqueeze(-1);  post_57 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_14, out_27, unsqueeze, comb_28);  x_14 = out_27 = unsqueeze = comb_28 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_14_parameters_hc_ffn_fn_, l_self_modules_layers_modules_14_parameters_hc_ffn_scale_, l_self_modules_layers_modules_14_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_14_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_14_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_14_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_14_parameters_hc_ffn_base_ = l_self_modules_layers_modules_14_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_14_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_14_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.14.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_15_parameters_hc_attn_fn_, l_self_modules_layers_modules_15_parameters_hc_attn_scale_, l_self_modules_layers_modules_15_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_15_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_15_parameters_hc_attn_fn_ = l_self_modules_layers_modules_15_parameters_hc_attn_scale_ = l_self_modules_layers_modules_15_parameters_hc_attn_base_ = l_self_modules_layers_modules_15_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_31(torch.nn.Module):
        def forward(self, y_30: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_30, l_positions_, 'layers.15.attn');  y_30 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_32(torch.nn.Module):
        def forward(self, out_29: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_61: "f32[s72, 4]", x_15: "bf16[s72, 4096]", comb_30: "f32[s72, 4, 4]", l_self_modules_layers_modules_15_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_15_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_15_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_15_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_15_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_16_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_16_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_16_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_16_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_29)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_61.unsqueeze(-1);  post_61 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_15, out_29, unsqueeze, comb_30);  x_15 = out_29 = unsqueeze = comb_30 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_15_parameters_hc_ffn_fn_, l_self_modules_layers_modules_15_parameters_hc_ffn_scale_, l_self_modules_layers_modules_15_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_15_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_15_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_15_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_15_parameters_hc_ffn_base_ = l_self_modules_layers_modules_15_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_15_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_15_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.15.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_16_parameters_hc_attn_fn_, l_self_modules_layers_modules_16_parameters_hc_attn_scale_, l_self_modules_layers_modules_16_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_16_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_16_parameters_hc_attn_fn_ = l_self_modules_layers_modules_16_parameters_hc_attn_scale_ = l_self_modules_layers_modules_16_parameters_hc_attn_base_ = l_self_modules_layers_modules_16_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_33(torch.nn.Module):
        def forward(self, y_32: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_32, l_positions_, 'layers.16.attn');  y_32 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_34(torch.nn.Module):
        def forward(self, out_31: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_65: "f32[s72, 4]", x_16: "bf16[s72, 4096]", comb_32: "f32[s72, 4, 4]", l_self_modules_layers_modules_16_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_16_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_16_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_16_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_16_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_17_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_17_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_17_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_17_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_31)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_65.unsqueeze(-1);  post_65 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_16, out_31, unsqueeze, comb_32);  x_16 = out_31 = unsqueeze = comb_32 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_16_parameters_hc_ffn_fn_, l_self_modules_layers_modules_16_parameters_hc_ffn_scale_, l_self_modules_layers_modules_16_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_16_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_16_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_16_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_16_parameters_hc_ffn_base_ = l_self_modules_layers_modules_16_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_16_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_16_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.16.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_17_parameters_hc_attn_fn_, l_self_modules_layers_modules_17_parameters_hc_attn_scale_, l_self_modules_layers_modules_17_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_17_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_17_parameters_hc_attn_fn_ = l_self_modules_layers_modules_17_parameters_hc_attn_scale_ = l_self_modules_layers_modules_17_parameters_hc_attn_base_ = l_self_modules_layers_modules_17_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_35(torch.nn.Module):
        def forward(self, y_34: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_34, l_positions_, 'layers.17.attn');  y_34 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_36(torch.nn.Module):
        def forward(self, out_33: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_69: "f32[s72, 4]", x_17: "bf16[s72, 4096]", comb_34: "f32[s72, 4, 4]", l_self_modules_layers_modules_17_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_17_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_17_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_17_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_17_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_18_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_18_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_18_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_18_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_33)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_69.unsqueeze(-1);  post_69 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_17, out_33, unsqueeze, comb_34);  x_17 = out_33 = unsqueeze = comb_34 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_17_parameters_hc_ffn_fn_, l_self_modules_layers_modules_17_parameters_hc_ffn_scale_, l_self_modules_layers_modules_17_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_17_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_17_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_17_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_17_parameters_hc_ffn_base_ = l_self_modules_layers_modules_17_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_17_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_17_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.17.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_18_parameters_hc_attn_fn_, l_self_modules_layers_modules_18_parameters_hc_attn_scale_, l_self_modules_layers_modules_18_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_18_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_18_parameters_hc_attn_fn_ = l_self_modules_layers_modules_18_parameters_hc_attn_scale_ = l_self_modules_layers_modules_18_parameters_hc_attn_base_ = l_self_modules_layers_modules_18_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_37(torch.nn.Module):
        def forward(self, y_36: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_36, l_positions_, 'layers.18.attn');  y_36 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_38(torch.nn.Module):
        def forward(self, out_35: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_73: "f32[s72, 4]", x_18: "bf16[s72, 4096]", comb_36: "f32[s72, 4, 4]", l_self_modules_layers_modules_18_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_18_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_18_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_18_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_18_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_19_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_19_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_19_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_19_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_35)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_73.unsqueeze(-1);  post_73 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_18, out_35, unsqueeze, comb_36);  x_18 = out_35 = unsqueeze = comb_36 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_18_parameters_hc_ffn_fn_, l_self_modules_layers_modules_18_parameters_hc_ffn_scale_, l_self_modules_layers_modules_18_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_18_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_18_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_18_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_18_parameters_hc_ffn_base_ = l_self_modules_layers_modules_18_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_18_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_18_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.18.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_19_parameters_hc_attn_fn_, l_self_modules_layers_modules_19_parameters_hc_attn_scale_, l_self_modules_layers_modules_19_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_19_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_19_parameters_hc_attn_fn_ = l_self_modules_layers_modules_19_parameters_hc_attn_scale_ = l_self_modules_layers_modules_19_parameters_hc_attn_base_ = l_self_modules_layers_modules_19_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_39(torch.nn.Module):
        def forward(self, y_38: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_38, l_positions_, 'layers.19.attn');  y_38 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_40(torch.nn.Module):
        def forward(self, out_37: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_77: "f32[s72, 4]", x_19: "bf16[s72, 4096]", comb_38: "f32[s72, 4, 4]", l_self_modules_layers_modules_19_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_19_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_19_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_19_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_19_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_20_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_20_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_20_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_20_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_37)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_77.unsqueeze(-1);  post_77 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_19, out_37, unsqueeze, comb_38);  x_19 = out_37 = unsqueeze = comb_38 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_19_parameters_hc_ffn_fn_, l_self_modules_layers_modules_19_parameters_hc_ffn_scale_, l_self_modules_layers_modules_19_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_19_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_19_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_19_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_19_parameters_hc_ffn_base_ = l_self_modules_layers_modules_19_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_19_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_19_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.19.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_20_parameters_hc_attn_fn_, l_self_modules_layers_modules_20_parameters_hc_attn_scale_, l_self_modules_layers_modules_20_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_20_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_20_parameters_hc_attn_fn_ = l_self_modules_layers_modules_20_parameters_hc_attn_scale_ = l_self_modules_layers_modules_20_parameters_hc_attn_base_ = l_self_modules_layers_modules_20_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_41(torch.nn.Module):
        def forward(self, y_40: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_40, l_positions_, 'layers.20.attn');  y_40 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_42(torch.nn.Module):
        def forward(self, out_39: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_81: "f32[s72, 4]", x_20: "bf16[s72, 4096]", comb_40: "f32[s72, 4, 4]", l_self_modules_layers_modules_20_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_20_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_20_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_20_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_20_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_21_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_21_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_21_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_21_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_39)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_81.unsqueeze(-1);  post_81 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_20, out_39, unsqueeze, comb_40);  x_20 = out_39 = unsqueeze = comb_40 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_20_parameters_hc_ffn_fn_, l_self_modules_layers_modules_20_parameters_hc_ffn_scale_, l_self_modules_layers_modules_20_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_20_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_20_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_20_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_20_parameters_hc_ffn_base_ = l_self_modules_layers_modules_20_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_20_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_20_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.20.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_21_parameters_hc_attn_fn_, l_self_modules_layers_modules_21_parameters_hc_attn_scale_, l_self_modules_layers_modules_21_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_21_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_21_parameters_hc_attn_fn_ = l_self_modules_layers_modules_21_parameters_hc_attn_scale_ = l_self_modules_layers_modules_21_parameters_hc_attn_base_ = l_self_modules_layers_modules_21_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_43(torch.nn.Module):
        def forward(self, y_42: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_42, l_positions_, 'layers.21.attn');  y_42 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_44(torch.nn.Module):
        def forward(self, out_41: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_85: "f32[s72, 4]", x_21: "bf16[s72, 4096]", comb_42: "f32[s72, 4, 4]", l_self_modules_layers_modules_21_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_21_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_21_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_21_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_21_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_22_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_22_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_22_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_22_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_41)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_85.unsqueeze(-1);  post_85 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_21, out_41, unsqueeze, comb_42);  x_21 = out_41 = unsqueeze = comb_42 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_21_parameters_hc_ffn_fn_, l_self_modules_layers_modules_21_parameters_hc_ffn_scale_, l_self_modules_layers_modules_21_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_21_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_21_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_21_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_21_parameters_hc_ffn_base_ = l_self_modules_layers_modules_21_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_21_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_21_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.21.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_22_parameters_hc_attn_fn_, l_self_modules_layers_modules_22_parameters_hc_attn_scale_, l_self_modules_layers_modules_22_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_22_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_22_parameters_hc_attn_fn_ = l_self_modules_layers_modules_22_parameters_hc_attn_scale_ = l_self_modules_layers_modules_22_parameters_hc_attn_base_ = l_self_modules_layers_modules_22_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_45(torch.nn.Module):
        def forward(self, y_44: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_44, l_positions_, 'layers.22.attn');  y_44 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_46(torch.nn.Module):
        def forward(self, out_43: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_89: "f32[s72, 4]", x_22: "bf16[s72, 4096]", comb_44: "f32[s72, 4, 4]", l_self_modules_layers_modules_22_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_22_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_22_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_22_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_22_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_23_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_23_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_23_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_23_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_43)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_89.unsqueeze(-1);  post_89 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_22, out_43, unsqueeze, comb_44);  x_22 = out_43 = unsqueeze = comb_44 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_22_parameters_hc_ffn_fn_, l_self_modules_layers_modules_22_parameters_hc_ffn_scale_, l_self_modules_layers_modules_22_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_22_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_22_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_22_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_22_parameters_hc_ffn_base_ = l_self_modules_layers_modules_22_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_22_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_22_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.22.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_23_parameters_hc_attn_fn_, l_self_modules_layers_modules_23_parameters_hc_attn_scale_, l_self_modules_layers_modules_23_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_23_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_23_parameters_hc_attn_fn_ = l_self_modules_layers_modules_23_parameters_hc_attn_scale_ = l_self_modules_layers_modules_23_parameters_hc_attn_base_ = l_self_modules_layers_modules_23_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_47(torch.nn.Module):
        def forward(self, y_46: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_46, l_positions_, 'layers.23.attn');  y_46 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_48(torch.nn.Module):
        def forward(self, out_45: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_93: "f32[s72, 4]", x_23: "bf16[s72, 4096]", comb_46: "f32[s72, 4, 4]", l_self_modules_layers_modules_23_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_23_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_23_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_23_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_23_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_24_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_24_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_24_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_24_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_45)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_93.unsqueeze(-1);  post_93 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_23, out_45, unsqueeze, comb_46);  x_23 = out_45 = unsqueeze = comb_46 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_23_parameters_hc_ffn_fn_, l_self_modules_layers_modules_23_parameters_hc_ffn_scale_, l_self_modules_layers_modules_23_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_23_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_23_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_23_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_23_parameters_hc_ffn_base_ = l_self_modules_layers_modules_23_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_23_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_23_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.23.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_24_parameters_hc_attn_fn_, l_self_modules_layers_modules_24_parameters_hc_attn_scale_, l_self_modules_layers_modules_24_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_24_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_24_parameters_hc_attn_fn_ = l_self_modules_layers_modules_24_parameters_hc_attn_scale_ = l_self_modules_layers_modules_24_parameters_hc_attn_base_ = l_self_modules_layers_modules_24_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_49(torch.nn.Module):
        def forward(self, y_48: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_48, l_positions_, 'layers.24.attn');  y_48 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_50(torch.nn.Module):
        def forward(self, out_47: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_97: "f32[s72, 4]", x_24: "bf16[s72, 4096]", comb_48: "f32[s72, 4, 4]", l_self_modules_layers_modules_24_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_24_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_24_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_24_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_24_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_25_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_25_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_25_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_25_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_47)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_97.unsqueeze(-1);  post_97 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_24, out_47, unsqueeze, comb_48);  x_24 = out_47 = unsqueeze = comb_48 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_24_parameters_hc_ffn_fn_, l_self_modules_layers_modules_24_parameters_hc_ffn_scale_, l_self_modules_layers_modules_24_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_24_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_24_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_24_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_24_parameters_hc_ffn_base_ = l_self_modules_layers_modules_24_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_24_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_24_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.24.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_25_parameters_hc_attn_fn_, l_self_modules_layers_modules_25_parameters_hc_attn_scale_, l_self_modules_layers_modules_25_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_25_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_25_parameters_hc_attn_fn_ = l_self_modules_layers_modules_25_parameters_hc_attn_scale_ = l_self_modules_layers_modules_25_parameters_hc_attn_base_ = l_self_modules_layers_modules_25_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_51(torch.nn.Module):
        def forward(self, y_50: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_50, l_positions_, 'layers.25.attn');  y_50 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_52(torch.nn.Module):
        def forward(self, out_49: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_101: "f32[s72, 4]", x_25: "bf16[s72, 4096]", comb_50: "f32[s72, 4, 4]", l_self_modules_layers_modules_25_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_25_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_25_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_25_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_25_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_26_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_26_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_26_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_26_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_49)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_101.unsqueeze(-1);  post_101 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_25, out_49, unsqueeze, comb_50);  x_25 = out_49 = unsqueeze = comb_50 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_25_parameters_hc_ffn_fn_, l_self_modules_layers_modules_25_parameters_hc_ffn_scale_, l_self_modules_layers_modules_25_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_25_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_25_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_25_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_25_parameters_hc_ffn_base_ = l_self_modules_layers_modules_25_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_25_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_25_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.25.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_26_parameters_hc_attn_fn_, l_self_modules_layers_modules_26_parameters_hc_attn_scale_, l_self_modules_layers_modules_26_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_26_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_26_parameters_hc_attn_fn_ = l_self_modules_layers_modules_26_parameters_hc_attn_scale_ = l_self_modules_layers_modules_26_parameters_hc_attn_base_ = l_self_modules_layers_modules_26_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_53(torch.nn.Module):
        def forward(self, y_52: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_52, l_positions_, 'layers.26.attn');  y_52 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_54(torch.nn.Module):
        def forward(self, out_51: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_105: "f32[s72, 4]", x_26: "bf16[s72, 4096]", comb_52: "f32[s72, 4, 4]", l_self_modules_layers_modules_26_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_26_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_26_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_26_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_26_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_27_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_27_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_27_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_27_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_51)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_105.unsqueeze(-1);  post_105 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_26, out_51, unsqueeze, comb_52);  x_26 = out_51 = unsqueeze = comb_52 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_26_parameters_hc_ffn_fn_, l_self_modules_layers_modules_26_parameters_hc_ffn_scale_, l_self_modules_layers_modules_26_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_26_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_26_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_26_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_26_parameters_hc_ffn_base_ = l_self_modules_layers_modules_26_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_26_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_26_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.26.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_27_parameters_hc_attn_fn_, l_self_modules_layers_modules_27_parameters_hc_attn_scale_, l_self_modules_layers_modules_27_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_27_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_27_parameters_hc_attn_fn_ = l_self_modules_layers_modules_27_parameters_hc_attn_scale_ = l_self_modules_layers_modules_27_parameters_hc_attn_base_ = l_self_modules_layers_modules_27_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_55(torch.nn.Module):
        def forward(self, y_54: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_54, l_positions_, 'layers.27.attn');  y_54 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_56(torch.nn.Module):
        def forward(self, out_53: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_109: "f32[s72, 4]", x_27: "bf16[s72, 4096]", comb_54: "f32[s72, 4, 4]", l_self_modules_layers_modules_27_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_27_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_27_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_27_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_27_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_28_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_28_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_28_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_28_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_53)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_109.unsqueeze(-1);  post_109 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_27, out_53, unsqueeze, comb_54);  x_27 = out_53 = unsqueeze = comb_54 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_27_parameters_hc_ffn_fn_, l_self_modules_layers_modules_27_parameters_hc_ffn_scale_, l_self_modules_layers_modules_27_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_27_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_27_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_27_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_27_parameters_hc_ffn_base_ = l_self_modules_layers_modules_27_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_27_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_27_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.27.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_28_parameters_hc_attn_fn_, l_self_modules_layers_modules_28_parameters_hc_attn_scale_, l_self_modules_layers_modules_28_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_28_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_28_parameters_hc_attn_fn_ = l_self_modules_layers_modules_28_parameters_hc_attn_scale_ = l_self_modules_layers_modules_28_parameters_hc_attn_base_ = l_self_modules_layers_modules_28_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_57(torch.nn.Module):
        def forward(self, y_56: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_56, l_positions_, 'layers.28.attn');  y_56 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_58(torch.nn.Module):
        def forward(self, out_55: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_113: "f32[s72, 4]", x_28: "bf16[s72, 4096]", comb_56: "f32[s72, 4, 4]", l_self_modules_layers_modules_28_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_28_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_28_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_28_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_28_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_29_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_29_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_29_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_29_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_55)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_113.unsqueeze(-1);  post_113 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_28, out_55, unsqueeze, comb_56);  x_28 = out_55 = unsqueeze = comb_56 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_28_parameters_hc_ffn_fn_, l_self_modules_layers_modules_28_parameters_hc_ffn_scale_, l_self_modules_layers_modules_28_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_28_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_28_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_28_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_28_parameters_hc_ffn_base_ = l_self_modules_layers_modules_28_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_28_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_28_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.28.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_29_parameters_hc_attn_fn_, l_self_modules_layers_modules_29_parameters_hc_attn_scale_, l_self_modules_layers_modules_29_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_29_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_29_parameters_hc_attn_fn_ = l_self_modules_layers_modules_29_parameters_hc_attn_scale_ = l_self_modules_layers_modules_29_parameters_hc_attn_base_ = l_self_modules_layers_modules_29_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_59(torch.nn.Module):
        def forward(self, y_58: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_58, l_positions_, 'layers.29.attn');  y_58 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_60(torch.nn.Module):
        def forward(self, out_57: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_117: "f32[s72, 4]", x_29: "bf16[s72, 4096]", comb_58: "f32[s72, 4, 4]", l_self_modules_layers_modules_29_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_29_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_29_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_29_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_29_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_30_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_30_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_30_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_30_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_57)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_117.unsqueeze(-1);  post_117 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_29, out_57, unsqueeze, comb_58);  x_29 = out_57 = unsqueeze = comb_58 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_29_parameters_hc_ffn_fn_, l_self_modules_layers_modules_29_parameters_hc_ffn_scale_, l_self_modules_layers_modules_29_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_29_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_29_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_29_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_29_parameters_hc_ffn_base_ = l_self_modules_layers_modules_29_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_29_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_29_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.29.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_30_parameters_hc_attn_fn_, l_self_modules_layers_modules_30_parameters_hc_attn_scale_, l_self_modules_layers_modules_30_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_30_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_30_parameters_hc_attn_fn_ = l_self_modules_layers_modules_30_parameters_hc_attn_scale_ = l_self_modules_layers_modules_30_parameters_hc_attn_base_ = l_self_modules_layers_modules_30_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_61(torch.nn.Module):
        def forward(self, y_60: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_60, l_positions_, 'layers.30.attn');  y_60 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_62(torch.nn.Module):
        def forward(self, out_59: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_121: "f32[s72, 4]", x_30: "bf16[s72, 4096]", comb_60: "f32[s72, 4, 4]", l_self_modules_layers_modules_30_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_30_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_30_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_30_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_30_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_31_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_31_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_31_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_31_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_59)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_121.unsqueeze(-1);  post_121 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_30, out_59, unsqueeze, comb_60);  x_30 = out_59 = unsqueeze = comb_60 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_30_parameters_hc_ffn_fn_, l_self_modules_layers_modules_30_parameters_hc_ffn_scale_, l_self_modules_layers_modules_30_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_30_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_30_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_30_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_30_parameters_hc_ffn_base_ = l_self_modules_layers_modules_30_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_30_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_30_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.30.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_31_parameters_hc_attn_fn_, l_self_modules_layers_modules_31_parameters_hc_attn_scale_, l_self_modules_layers_modules_31_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_31_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_31_parameters_hc_attn_fn_ = l_self_modules_layers_modules_31_parameters_hc_attn_scale_ = l_self_modules_layers_modules_31_parameters_hc_attn_base_ = l_self_modules_layers_modules_31_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_63(torch.nn.Module):
        def forward(self, y_62: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_62, l_positions_, 'layers.31.attn');  y_62 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_64(torch.nn.Module):
        def forward(self, out_61: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_125: "f32[s72, 4]", x_31: "bf16[s72, 4096]", comb_62: "f32[s72, 4, 4]", l_self_modules_layers_modules_31_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_31_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_31_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_31_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_31_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_32_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_32_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_32_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_32_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_61)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_125.unsqueeze(-1);  post_125 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_31, out_61, unsqueeze, comb_62);  x_31 = out_61 = unsqueeze = comb_62 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_31_parameters_hc_ffn_fn_, l_self_modules_layers_modules_31_parameters_hc_ffn_scale_, l_self_modules_layers_modules_31_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_31_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_31_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_31_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_31_parameters_hc_ffn_base_ = l_self_modules_layers_modules_31_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_31_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_31_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.31.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_32_parameters_hc_attn_fn_, l_self_modules_layers_modules_32_parameters_hc_attn_scale_, l_self_modules_layers_modules_32_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_32_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_32_parameters_hc_attn_fn_ = l_self_modules_layers_modules_32_parameters_hc_attn_scale_ = l_self_modules_layers_modules_32_parameters_hc_attn_base_ = l_self_modules_layers_modules_32_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_65(torch.nn.Module):
        def forward(self, y_64: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_64, l_positions_, 'layers.32.attn');  y_64 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_66(torch.nn.Module):
        def forward(self, out_63: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_129: "f32[s72, 4]", x_32: "bf16[s72, 4096]", comb_64: "f32[s72, 4, 4]", l_self_modules_layers_modules_32_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_32_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_32_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_32_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_32_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_33_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_33_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_33_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_33_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_63)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_129.unsqueeze(-1);  post_129 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_32, out_63, unsqueeze, comb_64);  x_32 = out_63 = unsqueeze = comb_64 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_32_parameters_hc_ffn_fn_, l_self_modules_layers_modules_32_parameters_hc_ffn_scale_, l_self_modules_layers_modules_32_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_32_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_32_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_32_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_32_parameters_hc_ffn_base_ = l_self_modules_layers_modules_32_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_32_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_32_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.32.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_33_parameters_hc_attn_fn_, l_self_modules_layers_modules_33_parameters_hc_attn_scale_, l_self_modules_layers_modules_33_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_33_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_33_parameters_hc_attn_fn_ = l_self_modules_layers_modules_33_parameters_hc_attn_scale_ = l_self_modules_layers_modules_33_parameters_hc_attn_base_ = l_self_modules_layers_modules_33_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_67(torch.nn.Module):
        def forward(self, y_66: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_66, l_positions_, 'layers.33.attn');  y_66 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_68(torch.nn.Module):
        def forward(self, out_65: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_133: "f32[s72, 4]", x_33: "bf16[s72, 4096]", comb_66: "f32[s72, 4, 4]", l_self_modules_layers_modules_33_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_33_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_33_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_33_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_33_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_34_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_34_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_34_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_34_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_65)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_133.unsqueeze(-1);  post_133 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_33, out_65, unsqueeze, comb_66);  x_33 = out_65 = unsqueeze = comb_66 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_33_parameters_hc_ffn_fn_, l_self_modules_layers_modules_33_parameters_hc_ffn_scale_, l_self_modules_layers_modules_33_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_33_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_33_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_33_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_33_parameters_hc_ffn_base_ = l_self_modules_layers_modules_33_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_33_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_33_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.33.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_34_parameters_hc_attn_fn_, l_self_modules_layers_modules_34_parameters_hc_attn_scale_, l_self_modules_layers_modules_34_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_34_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_34_parameters_hc_attn_fn_ = l_self_modules_layers_modules_34_parameters_hc_attn_scale_ = l_self_modules_layers_modules_34_parameters_hc_attn_base_ = l_self_modules_layers_modules_34_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_69(torch.nn.Module):
        def forward(self, y_68: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_68, l_positions_, 'layers.34.attn');  y_68 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_70(torch.nn.Module):
        def forward(self, out_67: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_137: "f32[s72, 4]", x_34: "bf16[s72, 4096]", comb_68: "f32[s72, 4, 4]", l_self_modules_layers_modules_34_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_34_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_34_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_34_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_34_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_35_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_35_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_35_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_35_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_67)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_137.unsqueeze(-1);  post_137 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_34, out_67, unsqueeze, comb_68);  x_34 = out_67 = unsqueeze = comb_68 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_34_parameters_hc_ffn_fn_, l_self_modules_layers_modules_34_parameters_hc_ffn_scale_, l_self_modules_layers_modules_34_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_34_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_34_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_34_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_34_parameters_hc_ffn_base_ = l_self_modules_layers_modules_34_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_34_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_34_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.34.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_35_parameters_hc_attn_fn_, l_self_modules_layers_modules_35_parameters_hc_attn_scale_, l_self_modules_layers_modules_35_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_35_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_35_parameters_hc_attn_fn_ = l_self_modules_layers_modules_35_parameters_hc_attn_scale_ = l_self_modules_layers_modules_35_parameters_hc_attn_base_ = l_self_modules_layers_modules_35_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_71(torch.nn.Module):
        def forward(self, y_70: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_70, l_positions_, 'layers.35.attn');  y_70 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_72(torch.nn.Module):
        def forward(self, out_69: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_141: "f32[s72, 4]", x_35: "bf16[s72, 4096]", comb_70: "f32[s72, 4, 4]", l_self_modules_layers_modules_35_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_35_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_35_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_35_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_35_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_36_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_36_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_36_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_36_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_69)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_141.unsqueeze(-1);  post_141 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_35, out_69, unsqueeze, comb_70);  x_35 = out_69 = unsqueeze = comb_70 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_35_parameters_hc_ffn_fn_, l_self_modules_layers_modules_35_parameters_hc_ffn_scale_, l_self_modules_layers_modules_35_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_35_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_35_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_35_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_35_parameters_hc_ffn_base_ = l_self_modules_layers_modules_35_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_35_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_35_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.35.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_36_parameters_hc_attn_fn_, l_self_modules_layers_modules_36_parameters_hc_attn_scale_, l_self_modules_layers_modules_36_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_36_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_36_parameters_hc_attn_fn_ = l_self_modules_layers_modules_36_parameters_hc_attn_scale_ = l_self_modules_layers_modules_36_parameters_hc_attn_base_ = l_self_modules_layers_modules_36_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_73(torch.nn.Module):
        def forward(self, y_72: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_72, l_positions_, 'layers.36.attn');  y_72 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_74(torch.nn.Module):
        def forward(self, out_71: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_145: "f32[s72, 4]", x_36: "bf16[s72, 4096]", comb_72: "f32[s72, 4, 4]", l_self_modules_layers_modules_36_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_36_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_36_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_36_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_36_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_37_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_37_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_37_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_37_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_71)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_145.unsqueeze(-1);  post_145 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_36, out_71, unsqueeze, comb_72);  x_36 = out_71 = unsqueeze = comb_72 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_36_parameters_hc_ffn_fn_, l_self_modules_layers_modules_36_parameters_hc_ffn_scale_, l_self_modules_layers_modules_36_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_36_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_36_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_36_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_36_parameters_hc_ffn_base_ = l_self_modules_layers_modules_36_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_36_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_36_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.36.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_37_parameters_hc_attn_fn_, l_self_modules_layers_modules_37_parameters_hc_attn_scale_, l_self_modules_layers_modules_37_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_37_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_37_parameters_hc_attn_fn_ = l_self_modules_layers_modules_37_parameters_hc_attn_scale_ = l_self_modules_layers_modules_37_parameters_hc_attn_base_ = l_self_modules_layers_modules_37_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_75(torch.nn.Module):
        def forward(self, y_74: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_74, l_positions_, 'layers.37.attn');  y_74 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_76(torch.nn.Module):
        def forward(self, out_73: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_149: "f32[s72, 4]", x_37: "bf16[s72, 4096]", comb_74: "f32[s72, 4, 4]", l_self_modules_layers_modules_37_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_37_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_37_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_37_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_37_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_38_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_38_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_38_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_38_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_73)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_149.unsqueeze(-1);  post_149 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_37, out_73, unsqueeze, comb_74);  x_37 = out_73 = unsqueeze = comb_74 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_37_parameters_hc_ffn_fn_, l_self_modules_layers_modules_37_parameters_hc_ffn_scale_, l_self_modules_layers_modules_37_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_37_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_37_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_37_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_37_parameters_hc_ffn_base_ = l_self_modules_layers_modules_37_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_37_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_37_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.37.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_38_parameters_hc_attn_fn_, l_self_modules_layers_modules_38_parameters_hc_attn_scale_, l_self_modules_layers_modules_38_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_38_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_38_parameters_hc_attn_fn_ = l_self_modules_layers_modules_38_parameters_hc_attn_scale_ = l_self_modules_layers_modules_38_parameters_hc_attn_base_ = l_self_modules_layers_modules_38_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_77(torch.nn.Module):
        def forward(self, y_76: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_76, l_positions_, 'layers.38.attn');  y_76 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_78(torch.nn.Module):
        def forward(self, out_75: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_153: "f32[s72, 4]", x_38: "bf16[s72, 4096]", comb_76: "f32[s72, 4, 4]", l_self_modules_layers_modules_38_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_38_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_38_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_38_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_38_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_39_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_39_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_39_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_39_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_75)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_153.unsqueeze(-1);  post_153 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_38, out_75, unsqueeze, comb_76);  x_38 = out_75 = unsqueeze = comb_76 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_38_parameters_hc_ffn_fn_, l_self_modules_layers_modules_38_parameters_hc_ffn_scale_, l_self_modules_layers_modules_38_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_38_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_38_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_38_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_38_parameters_hc_ffn_base_ = l_self_modules_layers_modules_38_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_38_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_38_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.38.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_39_parameters_hc_attn_fn_, l_self_modules_layers_modules_39_parameters_hc_attn_scale_, l_self_modules_layers_modules_39_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_39_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_39_parameters_hc_attn_fn_ = l_self_modules_layers_modules_39_parameters_hc_attn_scale_ = l_self_modules_layers_modules_39_parameters_hc_attn_base_ = l_self_modules_layers_modules_39_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_79(torch.nn.Module):
        def forward(self, y_78: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_78, l_positions_, 'layers.39.attn');  y_78 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_80(torch.nn.Module):
        def forward(self, out_77: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_157: "f32[s72, 4]", x_39: "bf16[s72, 4096]", comb_78: "f32[s72, 4, 4]", l_self_modules_layers_modules_39_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_39_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_39_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_39_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_39_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_40_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_40_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_40_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_40_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_77)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_157.unsqueeze(-1);  post_157 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_39, out_77, unsqueeze, comb_78);  x_39 = out_77 = unsqueeze = comb_78 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_39_parameters_hc_ffn_fn_, l_self_modules_layers_modules_39_parameters_hc_ffn_scale_, l_self_modules_layers_modules_39_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_39_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_39_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_39_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_39_parameters_hc_ffn_base_ = l_self_modules_layers_modules_39_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_39_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_39_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.39.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_40_parameters_hc_attn_fn_, l_self_modules_layers_modules_40_parameters_hc_attn_scale_, l_self_modules_layers_modules_40_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_40_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_40_parameters_hc_attn_fn_ = l_self_modules_layers_modules_40_parameters_hc_attn_scale_ = l_self_modules_layers_modules_40_parameters_hc_attn_base_ = l_self_modules_layers_modules_40_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_81(torch.nn.Module):
        def forward(self, y_80: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_80, l_positions_, 'layers.40.attn');  y_80 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_82(torch.nn.Module):
        def forward(self, out_79: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_161: "f32[s72, 4]", x_40: "bf16[s72, 4096]", comb_80: "f32[s72, 4, 4]", l_self_modules_layers_modules_40_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_40_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_40_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_40_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_40_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_41_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_41_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_41_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_41_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_79)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_161.unsqueeze(-1);  post_161 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_40, out_79, unsqueeze, comb_80);  x_40 = out_79 = unsqueeze = comb_80 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_40_parameters_hc_ffn_fn_, l_self_modules_layers_modules_40_parameters_hc_ffn_scale_, l_self_modules_layers_modules_40_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_40_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_40_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_40_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_40_parameters_hc_ffn_base_ = l_self_modules_layers_modules_40_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_40_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_40_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.40.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_41_parameters_hc_attn_fn_, l_self_modules_layers_modules_41_parameters_hc_attn_scale_, l_self_modules_layers_modules_41_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_41_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_41_parameters_hc_attn_fn_ = l_self_modules_layers_modules_41_parameters_hc_attn_scale_ = l_self_modules_layers_modules_41_parameters_hc_attn_base_ = l_self_modules_layers_modules_41_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_83(torch.nn.Module):
        def forward(self, y_82: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_82, l_positions_, 'layers.41.attn');  y_82 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_84(torch.nn.Module):
        def forward(self, out_81: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_165: "f32[s72, 4]", x_41: "bf16[s72, 4096]", comb_82: "f32[s72, 4, 4]", l_self_modules_layers_modules_41_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_41_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_41_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_41_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_41_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]", l_self_modules_layers_modules_42_parameters_hc_attn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_42_parameters_hc_attn_scale_: "f32[3]", l_self_modules_layers_modules_42_parameters_hc_attn_base_: "f32[24]", l_self_modules_layers_modules_42_modules_attn_norm_parameters_weight_: "bf16[4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_81)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_165.unsqueeze(-1);  post_165 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_41, out_81, unsqueeze, comb_82);  x_41 = out_81 = unsqueeze = comb_82 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_41_parameters_hc_ffn_fn_, l_self_modules_layers_modules_41_parameters_hc_ffn_scale_, l_self_modules_layers_modules_41_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_41_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_41_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_41_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_41_parameters_hc_ffn_base_ = l_self_modules_layers_modules_41_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_41_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_41_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.41.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre_1 = torch.ops.aiter.mhc_pre(empty_like_1, l_self_modules_layers_modules_42_parameters_hc_attn_fn_, l_self_modules_layers_modules_42_parameters_hc_attn_scale_, l_self_modules_layers_modules_42_parameters_hc_attn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_42_modules_attn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_42_parameters_hc_attn_fn_ = l_self_modules_layers_modules_42_parameters_hc_attn_scale_ = l_self_modules_layers_modules_42_parameters_hc_attn_base_ = l_self_modules_layers_modules_42_modules_attn_norm_parameters_weight_ = None
            getitem_3: "f32[s72, 4, 1]" = mhc_pre_1[0]
            getitem_4: "f32[s72, 4, 4]" = mhc_pre_1[1]
            getitem_5: "bf16[s72, 4096]" = mhc_pre_1[2];  mhc_pre_1 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze_1: "f32[s72, 4]" = getitem_3.squeeze(-1);  getitem_3 = None
            return (getitem_5, empty_like_1, squeeze_1, getitem_4)
            
    class submod_85(torch.nn.Module):
        def forward(self, y_84: "bf16[s72, 4096]", s72: "Sym(s72)", l_positions_: "i64[s80]", s80: "Sym(s80)"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            v4_attention_with_output: "bf16[s72, 4096]" = torch.ops.aiter.v4_attention_with_output(y_84, l_positions_, 'layers.42.attn');  y_84 = l_positions_ = None
            return v4_attention_with_output
            
    class submod_86(torch.nn.Module):
        def forward(self, out_83: "bf16[s72, 4, 4096]", s72: "Sym(s72)", post_169: "f32[s72, 4]", x_42: "bf16[s72, 4096]", comb_84: "f32[s72, 4, 4]", l_self_modules_layers_modules_42_parameters_hc_ffn_fn_: "f32[24, 16384]", l_self_modules_layers_modules_42_parameters_hc_ffn_scale_: "f32[3]", l_self_modules_layers_modules_42_parameters_hc_ffn_base_: "f32[24]", l_self_modules_layers_modules_42_modules_ffn_norm_parameters_weight_: "bf16[4096]", l_self_modules_layers_modules_42_modules_ffn_modules_gate_parameters_weight_: "bf16[256, 4096]"):
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like: "bf16[s72, 4, 4096]" = torch.empty_like(out_83)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze: "f32[s72, 4, 1]" = post_169.unsqueeze(-1);  post_169 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post = torch.ops.aiter.mhc_post(empty_like, x_42, out_83, unsqueeze, comb_84);  x_42 = out_83 = unsqueeze = comb_84 = mhc_post = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_pre = torch.ops.aiter.mhc_pre(empty_like, l_self_modules_layers_modules_42_parameters_hc_ffn_fn_, l_self_modules_layers_modules_42_parameters_hc_ffn_scale_, l_self_modules_layers_modules_42_parameters_hc_ffn_base_, 1e-06, 1e-06, 1e-06, 2.0, 20, l_self_modules_layers_modules_42_modules_ffn_norm_parameters_weight_, 1e-06);  l_self_modules_layers_modules_42_parameters_hc_ffn_fn_ = l_self_modules_layers_modules_42_parameters_hc_ffn_scale_ = l_self_modules_layers_modules_42_parameters_hc_ffn_base_ = l_self_modules_layers_modules_42_modules_ffn_norm_parameters_weight_ = None
            getitem: "f32[s72, 4, 1]" = mhc_pre[0]
            getitem_1: "f32[s72, 4, 4]" = mhc_pre[1]
            getitem_2: "bf16[s72, 4096]" = mhc_pre[2];  mhc_pre = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            squeeze: "f32[s72, 4]" = getitem.squeeze(-1);  getitem = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            gemm_a16w16: "bf16[s72, 256]" = torch.ops.aiter.gemm_a16w16(getitem_2, l_self_modules_layers_modules_42_modules_ffn_modules_gate_parameters_weight_, bias = None, otype = torch.bfloat16, scale_a = None, scale_b = None, scale_c = None);  l_self_modules_layers_modules_42_modules_ffn_modules_gate_parameters_weight_ = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            moe_forward: "bf16[s72, 4096]" = torch.ops.aiter.moe_forward(getitem_2, gemm_a16w16, 'layers.42.ffn.experts');  getitem_2 = gemm_a16w16 = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            all_reduce_: "bf16[s72, 4096]" = torch.ops.aiter.all_reduce_(moe_forward, group_name = 'tp:0', ca_use_new = True, ca_fp8_quant = False, prefill_support = False);  moe_forward = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            empty_like_1: "bf16[s72, 4, 4096]" = torch.empty_like(empty_like)
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            unsqueeze_1: "f32[s72, 4, 1]" = squeeze.unsqueeze(-1);  squeeze = None
            
             # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:103 in __torch_function__, code: return func(*args, **kwargs)
            mhc_post_1 = torch.ops.aiter.mhc_post(empty_like_1, all_reduce_, empty_like, unsqueeze_1, getitem_1);  all_reduce_ = empty_like = unsqueeze_1 = getitem_1 = mhc_post_1 = None
            return empty_like_1
            