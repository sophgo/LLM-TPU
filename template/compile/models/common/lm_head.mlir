#loc = loc(unknown)
module @lm_head attributes {module.FLOPs = 1090012676 : i64, module.chip = "ALL", module.platform = "TORCH", module.state = "TOP_F32", module.top_run_mode = "STATIC", module.weight_file = "lm_head_top_f32_all_weight.npz"} {
  func.func @main(%arg0: tensor<1x{hidden_size}xf32> loc(unknown)) -> tensor<1x{vocab_size}xf32> {
    %0 = "top.None"() : () -> none loc(#loc)
    %1 = "top.Input"(%arg0) {do_preprocess = true} : (tensor<1x{hidden_size}xf32>) -> tensor<1x{hidden_size}xf32> loc(#loc1)
    %2 = "top.Weight"() : () -> tensor<1x{hidden_size}xf32> loc(#loc2)
    %3 = "top.RMSNorm"(%1, %2) {eps = 9.9999999747524271E-7 : f64} : (tensor<1x{hidden_size}xf32>, tensor<1x{hidden_size}xf32>) -> tensor<1x{hidden_size}xf32> loc(#loc3)
    %4 = "top.Weight"() : () -> tensor<{hidden_size}x{vocab_size}xf32> loc(#loc4)
    %5 = "top.MatMul"(%3, %4, %0) {do_relu = false, hdim_is_batch = false, keep_dims = true, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f64, right_transpose = false} : (tensor<1x{hidden_size}xf32>, tensor<{hidden_size}x{vocab_size}xf32>, none) -> tensor<1x{vocab_size}xf32> loc(#loc5)
    return %5 : tensor<1x{vocab_size}xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("hidden_states.1")
#loc2 = loc("weight.2")
#loc3 = loc("23")
#loc4 = loc("26_filter")
#loc5 = loc("26")

