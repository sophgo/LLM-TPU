## cpp_demo_share_prompt

A long prompt can be converted into a KV cache, and every subsequent
question shares this KV cache. The input given by `--prompt` /
`--prompt_file` / `--media_path` is only prefilled to generate the kv cache
and states (no answer is generated). The demo then enters the interactive
chat loop, and every question is independently based on the shared prompt —
the kv cache and states are rolled back to the shared prompt's snapshot
before each question.

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

``` shell
./pipeline -m bmodel_path -c config \
    --prompt_file story.txt --media_path test.jpg
```

Any combination of `--prompt` / `--prompt_file` / `--media_path` can be
used as the shared prompt (at least one of them is required).

Notes:

- The bmodel must be compiled with `--use_history_kv` (history support).
  Make sure the shared prompt plus each question fits in `--seq_length` of
  `llm_convert.py`.
- Questions do not accumulate chat history; each question starts from the
  shared prompt's snapshot.
