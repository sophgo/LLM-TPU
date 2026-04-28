#!/usr/bin/env bash
#
# Quick-launch script for LLM-TPU demos.
#
# Usage:
#   ./run.sh --model <name>
#
# Run `./run.sh --help` for the list of supported demo names.

set -euo pipefail

# Mapping from short demo name -> directory under models/
declare -A MODEL_TO_DEMO=(
    ["qwen3"]="Qwen3"
    ["qwen2.5vl"]="Qwen2_5_VL"
    ["internvl3"]="InternVL3"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    local names
    names="$(printf '%s, ' "${!MODEL_TO_DEMO[@]}" | sed 's/, $//')"
    cat <<EOF
Usage: $(basename "$0") --model <name>

Options:
  --model <name>   Demo to run. Supported: ${names}
  -h, --help       Show this help message and exit.

Example:
  $(basename "$0") --model qwen2.5vl
EOF
}

err() { echo "Error: $*" >&2; }

parse_args() {
    model=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model)
                if [[ $# -lt 2 ]]; then
                    err "--model requires an argument."
                    usage >&2
                    exit 2
                fi
                model="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                err "Invalid option: $1"
                usage >&2
                exit 2
                ;;
        esac
    done
}

parse_args "$@"

if [[ -z "${model}" ]]; then
    err "Missing required option: --model"
    usage >&2
    exit 2
fi

if [[ -z "${MODEL_TO_DEMO[$model]+_}" ]]; then
    valid="$(printf '%s, ' "${!MODEL_TO_DEMO[@]}" | sed 's/, $//')"
    err "Unknown model '${model}'. Supported: ${valid}"
    exit 2
fi

demo_dir="${SCRIPT_DIR}/models/${MODEL_TO_DEMO[$model]}"
demo_script="${demo_dir}/run_demo.sh"

if [[ ! -d "${demo_dir}" ]]; then
    err "Demo directory not found: ${demo_dir}"
    exit 1
fi
if [[ ! -x "${demo_script}" ]]; then
    err "Demo script missing or not executable: ${demo_script}"
    exit 1
fi

echo ">>> Running demo: ${model} (${demo_dir})"
cd "${demo_dir}"
exec ./run_demo.sh
