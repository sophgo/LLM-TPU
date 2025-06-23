# Upload Tool

一个安全的SFTP目录上传工具，支持压缩传输和进度显示。

### 使用方式
```bash
pip install paramiko
```
```bash
python upload.py \
    --host <SFTP服务器地址> \
    --port <端口号> \
    --username <用户名> \
    --password <密码> \
    --remote_dir <远程目录> \
    --local_dir <本地目录>
```
### 参数说明

| 参数        | 必须 | 默认值 | 说明                      |
|-------------|------|--------|-------------------------|
| --host      | 是   | 无     | SFTP服务器域名/IP地址    |
| --port      | 否   | 22     | SFTP服务端口号          |
| --username  | 是   | 无     | 认证用户名              |
| --password  | 是   | 无     | 认证密码                |
| --remote_dir| 是   | 无     | 远程目标目录路径        |
| --local_dir | 是   | 无     | 要上传的本地目录路径    |

### 注意事项

1. ⚠️ **目录限制**：禁止上传包含以下子目录的目录：
   - `onnx`
   - `bmodel`

2. 🗑️ **临时文件**：生成的压缩包会在上传完成后自动删除