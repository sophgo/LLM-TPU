## python demo

```
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

### CLI demo
```bash
python3 pipeline.py --model_path your_bmodel_path -c config
```

### Programmatic (non-interactive) mode
通过 `-p/--prompt` 指定一次性问题，运行一次推理后退出，便于脚本调用：
```bash
python3 pipeline.py --model_path your_bmodel_path -c config -p "你好，请简单介绍一下你自己。"
```
