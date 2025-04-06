#loc = loc(unknown)
module @embedding attributes {module.FLOPs = 0 : i64, module.chip = "ALL", module.platform = "TORCH", module.state = "TOP_F32", module.top_run_mode = "STATIC", module.weight_file = "embedding_top_f32_all_weight.npz"} {
  func.func @main(%arg0: tensor<1x{seq_length}xsi32> loc(unknown)) -> tensor<1x{seq_length}x{hidden_size}xf32> {
    %0 = "top.Input"(%arg0) {do_preprocess = true} : (tensor<1x{seq_length}xsi32>) -> tensor<1x{seq_length}xf32> loc(#loc1)
    %1 = "top.Weight"() : () -> tensor<{vocab_size}x{hidden_size}xf32> loc(#loc2)
    %2 = "top.Gather"(%1, %0) {axis = 0 : si32, is_scalar = false, keepdims = true} : (tensor<{vocab_size}x{hidden_size}xf32>, tensor<1x{seq_length}xf32>) -> tensor<1x{seq_length}x{hidden_size}xf32> loc(#loc3)
    return %2 : tensor<1x{seq_length}x{hidden_size}xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("input_ids.1")
#loc2 = loc("weight.1")
#loc3 = loc("12")

