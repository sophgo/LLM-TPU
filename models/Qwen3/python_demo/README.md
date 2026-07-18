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
Specify a one-shot question with `-p/--prompt`; the program runs one inference and exits, making it easy to call from scripts:
```bash
python3 pipeline.py --model_path your_bmodel_path -c config -p "你好，请简单介绍一下你自己。"
```
