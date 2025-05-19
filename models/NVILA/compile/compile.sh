#!/bin/bash
pushd tmp

mkdir -p vit_bmodel
pushd vit_bmodel

model_transform.py \
  --model_name vit \
  --model_def ../onnx/vit.onnx \
  --mlir vit.mlir 

model_deploy.py \
  --mlir vit.mlir \
  --quantize F16 \
  --quant_output \
  --chip bm1684x \
  --model vit.bmodel

model_transform.py \
  --model_name projector \
  --model_def ../onnx/projector.onnx \
  --mlir projector.mlir 

model_deploy.py \
  --mlir projector.mlir \
  --quantize F16 \
  --quant_input \
  --quant_output \
  --chip bm1684x \
  --model projector.bmodel

model_transform.py \
  --model_name vitmm \
  --model_def ../onnx/vitmm.onnx \
  --mlir vitmm.mlir

model_deploy.py \
  --mlir vitmm.mlir \
  --quantize F16 \
  --quant_output \
  --chip bm1684x \
  --model vitmm.bmodel

popd
model_tool --combine llm*.bmodel vit_bmodel/vitmm.bmodel vit_bmodel/projector.bmodel vit_bmodel/vit.bmodel -o ../nvila-8b_w4f16_bm1684x.bmodel
model_tool --info ../nvila-8b_w4f16_bm1684x.bmodel > model.log
popd