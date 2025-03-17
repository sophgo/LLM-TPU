#!/bin/bash
set -ex

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_dir)      model_dir="$2"; shift 2 ;;
        --seq_length_list) seq_list="$2"; shift 2 ;;
        --quantize)       quantize="$2"; shift 2 ;;
        --tpu_mlir_path)  tpu_mlir_path="$2"; shift 2 ;;
        --host)           sftp_host="$2"; shift 2 ;;
        --port)           sftp_port="$2"; shift 2 ;;
        --username)       sftp_user="$2"; shift 2 ;;
        --password)       sftp_pass="$2"; shift 2 ;;
        --remote_dir)     remote_base="$2"; shift 2 ;;
        *)                echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Verify required parameters
if [[ -z "$model_dir" || -z "$seq_list" || -z "$quantize" || -z "$tpu_mlir_path" || -z "$sftp_host" || -z "$sftp_user" || -z "$sftp_pass" || -z "$remote_base" ]]; then
    echo "Missing required parameters!"
    exit 1
fi

install_package() {
    pip install einops transformers_stream_generator
}

install_package

# Configure logging with date-based filename
log_file="log_$(date +%Y%m%d).log"
exec > >(tee "$log_file") 2>&1

# Process sequence length list
IFS=',' read -ra SEQ_LENGTHS <<< "$seq_list"

# Main processing loop
find "$model_dir" -mindepth 1 -maxdepth 1 -type d | while read -r model_path; do
    model_name=$(basename "$model_path" | tr '[:upper:]' '[:lower:]')
    
    # Iterate through sequence lengths
    for seq_length in "${SEQ_LENGTHS[@]}"; do
        out_dir="${model_name}_${quantize}_s${seq_length}"
        
        # Execute model export
        echo "Processing model: $model_name with seq_length $seq_length"
        python model_export.py \
            --quantize "$quantize" \
            --tpu_mlir_path "$tpu_mlir_path" \
            --model_path "$model_path" \
            --seq_length "$seq_length" \
            --out_dir "$out_dir" \
            --compile_mode fast \
            --embedding_disk || { echo "Model export failed"; exit 1; }

        # Clean temporary directories
        echo "Cleaning intermediate files for $out_dir"
        (cd "$out_dir" && rm -rf onnx bmodel) || { echo "Failed to clean directories"; exit 1; }

        # Upload results
        echo "Uploading $out_dir to SFTP"
        python tools/upload.py \
            --host "$sftp_host" \
            --port "${sftp_port:-22}" \
            --username "$sftp_user" \
            --password "$sftp_pass" \
            --remote_dir "${remote_base}/${out_dir}" \
            --local_dir "$out_dir" || { echo "Upload failed"; exit 1; }
    done
done

echo "All tasks completed! Log saved to: $log_file"