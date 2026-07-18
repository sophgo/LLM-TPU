## Compilation method (without the bundled sophon-opencv)

``` shell
# Install opencv dependencies
sudo apt update
sudo apt install libopencv-dev

# Compile
mkdir build && cd build 
cmake .. && make
```


## Compilation method (with the bundled /opt/sophon/sophon-opencv-latest)

You need to modify this line in CMakeLists.txt as follows:
```cmake
set(SOPHON_OPENCV TRUE)
```

Then compile
``` shell
mkdir build && cd build 
cmake .. && make
```

## Run
./pipeline -m bmodel_path -c config
