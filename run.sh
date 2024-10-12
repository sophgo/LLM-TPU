#!/bin/bash
set -ex

# Args
parse_args() {
    while [[ $# -gt 0 ]]; do
        key="$1"

        case $key in
            --model)
                model="$2"
                shift 2
                ;;
            --arch)
                arch="$2"
                shift 2
                ;;
            *)
                echo "Invalid option: $key" >&2
                exit 1
                ;;
            :)
                echo "Option -$OPTARG requires an argument." >&2
                exit 1
                ;;
        esac
    done
}

# Mapping
declare -A model_to_demo=(
    ["chatglm2-6b"]="ChatGLM2"
    ["chatglm3-6b"]="ChatGLM3"
    ["llama2-7b"]="Llama2"
    ["llama3-7b"]="Llama3"
    ["qwen-7b"]="Qwen"
    ["qwen1.5-1.8b"]="Qwen1_5"
    ["qwen2.5-7b"]="Qwen2_5"
    ["wizardcoder-15b"]="WizardCoder"
    ["lwm-text-chat"]="LWM"
    ["internvl2-4b"]="InternVL2"
    ["minicpmv2_6"]="MiniCPM-V-2_6"
)

# Process Args
parse_args "$@"

# Check Version
compare_date="20240110"
if [ $arch == "pcie" ]; then
    extracted_date=$(cat /proc/bmsophon/driver_version | grep -o 'release date: [0-9]\{8\}' | grep -o '[0-9]\{8\}')

elif [ $arch = "soc" ]; then
    extracted_date_str=$(uname -a | grep -oP 'SMP \K[A-Za-z]+\s[A-Za-z]+\s\d+\s\d+:\d+:\d+\s[A-Za-z]+\s\d+' | sed 's/HKT //')
    extracted_date=$(date -d "$extracted_date_str" '+%Y%m%d')
fi
if [ "$extracted_date" -lt "$compare_date" ]; then
    >&2 echo -e "Your driver is \033[31moutdated\033[0m. Please update your driver."
    exit 1
else
    echo "Driver date is $extracted_date, which is up to date. Continuing..."
fi

# Function to validate model name
validate_model() {
    local model="$1"
    if [[ ! ${model_to_demo[$model]} ]]; then
        echo -e "Error: Invalid name $model, the input name must be \033[31m$(printf "%s|" "${!model_to_demo[@]}" | sed 's/|$//')\033[0m" >&2
        return 1
    fi
    return 0
}

# Check Model Name
validate_model "$model" || exit 1

# Compile
pushd "./models/${model_to_demo[$model]}"
./run_demo.sh
popd
