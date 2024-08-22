model_transform.py \
    --model_name intern_vit \
    --model_def ./vit/onnx/vision_transformer.onnx \
    --input_shapes [[1,3,448,448]] \
    --mlir intern_vit.mlir \

model_deploy.py \
    --mlir intern_vit.mlir \
    --quantize F16 \
    --processor bm1684x \
    --quant_output \
    --model intern_vit_1684x_f16.bmodel