transformers_version=$(grep '"transformers_version"' config.json | cut -d':' -f2 | tr -d ' ",')
pkg_path=$(pip show transformers | grep Location | cut -d ' ' -f2)
pkg_version=$(pip show transformers | grep Version | cut -d ' ' -f2)

if [ "$transformers_version" != "$pkg_version" ]; then
  echo -e "\e[31mError: transformers version in config.json ($transformers_version) does not match the installed version ($pkg_version).\e[0m"
  exit 1
fi

cp ${pkg_path}/transformers/models/qwen2/modeling_qwen2.py modeling_qwen2_backup.py 
sudo cp modeling_qwen2.py ${pkg_path}/transformers/models/qwen2/modeling_qwen2.py
