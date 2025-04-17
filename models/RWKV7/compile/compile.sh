rwkv()
{
  local type=$1
  model_transform.py \
    --model_name rwkv_forward_one \
    --model_def ../one/rwkv_forward_one.onnx \
    --mlir rwkv_forward_one.mlir

  model_deploy.py \
    --mlir rwkv_forward_one.mlir \
    --quantize $type \
    --chip bm1684x \
	  --model rwkv_forward_one.bmodel

  model_transform.py \
    --model_name rwkv_forward_seq \
    --model_def ../seq/rwkv_forward_seq.onnx \
    --mlir rwkv_forward_seq.mlir

  model_deploy.py \
    --mlir rwkv_forward_seq.mlir \
    --quantize $type \
    --chip bm1684x \
	  --model rwkv_forward_seq.bmodel
}

pushd tmp
mkdir -p bmodel
cd bmodel

rwkv F16
model_tool --combine rwkv_forward_seq.bmodel, rwkv_forward_one.bmodel -o ../rwkv7_f16.bmodel

popd