abnormal_path="/workspace/LLM-TPU/template/multi_stage_demo/test_abnormal" # change to your abnormal_path
bmodel_path="/workspace/LLM-TPU/template/multi_stage_demo/qwen2_vl_2b/bmodel/encrypted.bmodel" # change to your bmodel_path
embedding_bin_path="/workspace/LLM-TPU/template/multi_stage_demo/qwen2_vl_2b/embedding.bin" # change to your embedding_path

pushd ${abnormal_path}
touch embedding.bin.empty

# lora
# touch encrypted_lora_weights.bin.empty
cp ${embedding_bin_path} .
split -b 300M embedding.bin embedding.bin.split
dd if=embedding.bin of=embedding.bin.split0 bs=56 count=1
dd if=embedding.bin of=embedding.bin.split1 bs=64 count=1

# bmodel
cp ${bmodel_path} . 
touch encrypted.bmodel.empty
dd if=${bmodel_path} of=encrypted.bmodel.split0 bs=56 count=1
dd if=${bmodel_path} of=encrypted.bmodel.split1 bs=64 count=1
dd if=${bmodel_path} of=encrypted.bmodel.split2 bs=100 count=1
dd if=${bmodel_path} of=encrypted.bmodel.split3 bs=10000 count=1
dd if=${bmodel_path} of=encrypted.bmodel.split4 bs=1000000 count=1
dd if=${bmodel_path} of=encrypted.bmodel.split5 bs=100000000 count=1
popd
