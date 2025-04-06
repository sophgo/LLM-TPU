#loc = loc(unknown)
module @greedy_head attributes {module.FLOPs = 0 : i64, module.chip = "ALL", module.platform = "ONNX", module.state = "TOP_F32", module.top_run_mode = "STATIC", module.weight_file = "greedy_head_top_f32_all_weight.npz"} {
  func.func @main(%arg0: tensor<1x{vocab_size}xf32> loc(unknown)) -> tensor<1x1xf32> {
    %0 = "top.Input"(%arg0) {do_preprocess = true} : (tensor<1x{vocab_size}xf32>) -> tensor<1x{vocab_size}xf32> loc(#loc1)
    %values, %indices = "top.TopK"(%0) {K = 1 : i64, axis = 1 : i64, largest = true, replace_topk_indices = false, sorted = true} : (tensor<1x{vocab_size}xf32>) -> (tensor<1x1xf32>, tensor<1x1xf32>) loc(#loc4)
    return %indices : tensor<1x1xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("m_logits")
#loc2 = loc("/TopK_output_0_TopK")
#loc3 = loc("token_TopK")
#loc4 = loc(fused[#loc2, #loc3])

