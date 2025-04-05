import argparse
import getpass
import os
import tarfile
import tempfile
import paramiko
import sys
import time

def main():
    """Main entry point for the SFTP upload application."""
    parser = argparse.ArgumentParser(description='Secure SFTP directory upload with compression.')
    parser.add_argument('--host', required=True, help='SFTP server hostname')
    parser.add_argument('--port', type=int, default=22, help='SFTP server port (default: 22)')
    parser.add_argument('--username', required=True, help='SFTP username')
    parser.add_argument('--password', required=True, help='SFTP password')
    parser.add_argument('--remote_dir', required=True, help='Remote target directory')
    parser.add_argument('--local_dir', required=True, help='Local directory to upload')
    args = parser.parse_args()
    
    try:
        # Validate local directory existence
        if not os.path.isdir(args.local_dir):
            raise ValueError(f"Local directory does not exist: {args.local_dir}")
        
        # Check for restricted directories
        validate_directory_contents(args.local_dir)
        
        # Create compressed archive
        temp_path = create_compressed_archive(args.local_dir)
        
        # Perform SFTP transfer
        sftp_transfer(
            host=args.host,
            port=args.port,
            username=args.username,
            password=args.password,
            remote_dir=args.remote_dir,
            local_archive=temp_path
        )
    finally:
        # Clean up temporary files
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

class TransferProgress:
    def __init__(self, total_files=0, total_size=0, operation="Processing"):
        self.start_time = time.time()
        self.total_files = total_files
        self.total_size = total_size
        self.processed_files = 0
        self.transferred_bytes = 0
        self.operation = operation
        self.last_print = 0
        
    def print_progress(self, current, total=None):
        now = time.time()
        if total:
            percent = current / total * 100
            speed = current / (now - self.start_time) / 1024 if now > self.start_time else 0
            sys.stdout.write(
                f"\r{self.operation}: {percent:.1f}% | "
                f"{current/1024/1024:.1f}MB/{total/1024/1024:.1f}MB | "
                f"{speed:.1f}KB/s | "
                f"Elapsed: {int(now - self.start_time)}s"
            )
        else:
            sys.stdout.write(
                f"\r{self.operation}: {current}/{self.total_files} files "
                f"({self.processed_files/self.total_files*100:.1f}%)"
            )
        sys.stdout.flush()
        self.last_print = now

def validate_directory_contents(local_dir):
    """Validate directory contents against restricted patterns.
    
    Args:
        local_dir (str): Path to local directory
        
    Raises:
        RuntimeError: If restricted directories are found
    """
    restricted_dirs = {'onnx', 'bmodel'}
    found_dirs = [d for d in os.listdir(local_dir) 
                if os.path.isdir(os.path.join(local_dir, d)) and d in restricted_dirs]
    
    if found_dirs:
        raise RuntimeError(
            f"Directory contains restricted subdirectories: {', '.join(found_dirs)}. "
            "Upload aborted."
        )

def create_compressed_archive(source_dir):
    """Create compressed archive of the source directory.
    
    Args:
        source_dir (str): Path to directory to compress
        
    Returns:
        str: Path to created temporary archive file
        
    Raises:
        RuntimeError: If archive creation fails
    """
    try:
        total_files = 0
        total_size = 0
        for root, dirs, files in os.walk(source_dir):
            total_files += len(files)
            total_size += sum(os.path.getsize(os.path.join(root, f)) for f in files)

        progress = TransferProgress(total_files, operation="Compressing")
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.tar.gz', 
            delete=False
        )

        def _progress_callback(tarinfo):
            progress.processed_files += 1
            if time.time() - progress.last_print > 0.1 or progress.processed_files == progress.total_files:
                progress.print_progress(progress.processed_files)
            return tarinfo

        COMPRESS_LEVEL = 1
        BUF_SIZE = 256 * 1024
        with tarfile.open(temp_file.name, 'w:gz', compresslevel=COMPRESS_LEVEL, bufsize=BUF_SIZE) as tar:
            tar.add(source_dir, 
                   arcname=os.path.basename(os.path.normpath(source_dir)),
                   filter=_progress_callback)
            
        print(f"\nCompression complete: {temp_file.name}")
        return temp_file.name
    except Exception as e:
        if 'temp_file' in locals():
            os.remove(temp_file.name)
        raise RuntimeError(f"Archive creation failed: {str(e)}") from e

def sftp_transfer(host, port, username, password, remote_dir, local_archive):
    """Handle SFTP file transfer operations.
    
    Args:
        host (str): SFTP server hostname
        port (int): SFTP server port
        username (str): Authentication username
        password (str): Authentication password
        remote_dir (str): Target remote directory
        local_archive (str): Path to local archive file
        
    Raises:
        RuntimeError: If any SFTP operation fails
    """
    transport = None
    sftp = None
    
    try:
        transport = paramiko.Transport((host, port))
        transport.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        
        # 确保目录存在逻辑保持不变...
        
        # 新增上传进度回调
        file_size = os.path.getsize(local_archive)
        progress = TransferProgress(total_size=file_size, operation="Uploading")
        
        def _upload_callback(sent, total):
            if time.time() - progress.last_print > 0.3 or sent == total:
                progress.print_progress(sent, total)
        
        remote_path = f"{remote_dir}/{os.path.basename(local_archive)}"
        sftp.put(local_archive, remote_path, callback=_upload_callback)
        print("\nUpload completed")
        
    except Exception as e:
        raise RuntimeError(f"SFTP operation failed: {str(e)}") from e
    finally:
        # Ensure proper connection cleanup
        if sftp:
            sftp.close()
        if transport:
            transport.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Execution error: {str(e)}")
        exit(1)
