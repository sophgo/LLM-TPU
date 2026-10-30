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

Text-only chat (no media):
```shell
./pipeline -m your_model.bmodel -c ../config
```

Chat about an image or video by attaching `@<path>` inside the question:
```shell
./pipeline -m your_model.bmodel -c ../config
# then at the prompt:
#   Question: @../test.jpg Describe this image.
#   Question: @../test.mp4 What is happening in this video?
```

Multi-device (comma separated) and sampling:
```shell
./pipeline -m your_model.bmodel -c ../config -d 0,1 --do_sample
```

Enter `/clear` (or `/new`) to reset the conversation history, and `/exit`
(`/q`, `/quit`) to leave.
