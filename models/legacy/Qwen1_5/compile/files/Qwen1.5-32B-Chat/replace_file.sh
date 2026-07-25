pkg_path=$(pip show transformers | grep Location | cut -d ' ' -f2)
cp ${pkg_path}/transformers/models/qwen2/modeling_qwen2.py modeling_qwen2_backup.py 
sudo cp modeling_qwen2.py ${pkg_path}/transformers/models/qwen2/modeling_qwen2.py