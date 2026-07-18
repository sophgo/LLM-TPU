``` shell
# Install opencv dependencies
sudo apt update
sudo apt install libopencv-dev

# Compile
mkdir build && cd build 
cmake .. && make

# Run
./pipeline -m bmodel_path -c config
```
