# Upload Tool

A secure SFTP directory upload tool that supports compressed transfer and progress display.

### Usage
```bash
pip install paramiko
```
```bash
python upload.py \
    --host <SFTP server address> \
    --port <port number> \
    --username <username> \
    --password <password> \
    --remote_dir <remote directory> \
    --local_dir <local directory>
```
### Parameter Description

| Parameter    | Required | Default | Description                 |
|--------------|----------|---------|-----------------------------|
| --host       | Yes      | None    | SFTP server domain name/IP address |
| --port       | No       | 22      | SFTP server port number     |
| --username   | Yes      | None    | Authentication username     |
| --password   | Yes      | None    | Authentication password     |
| --remote_dir | Yes      | None    | Remote target directory path |
| --local_dir  | Yes      | None    | Local directory path to upload |

### Notes

1. ⚠️ **Directory restriction**: Do not upload directories containing the following subdirectories:
   - `onnx`
   - `bmodel`

2. 🗑️ **Temporary files**: The generated compressed package will be automatically deleted after the upload completes