#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
MODEL_DIR=$DIR/../models



# =================== 编译所有的demo代码 =====================
EXCLUDE_DIRS=(
    "./Qwen2/demo_parallel"  # Qwen2的demo_parallel需要单独编译
    "./Qwen1_5/demo_parallel"  # Qwen1_5的demo_parallel需要单独编译
    "./Qwen/demo_parallel" # Qwen的demo_parallel需要单独编译
    "./LWM/demo"
    "./VILA1_5/cpp_demo"
)

is_excluded() {
  local dir="$1"
  for ex in "${EXCLUDE_DIRS[@]}"; do
    # 统一成相对 MODEL_DIR 的路径形式（以 ./ 开头）
    if [[ "$dir" == "$ex" ]]; then
      return 0
    fi
  done
  return 1
}

TARGET_FILE="CMakeLists.txt"
pushd $MODEL_DIR

find . -type f -name "$TARGET_FILE" \
    -not -path "*/.*/*" \
    -not -path "*/build/*" \
    -not -path "*/third_party/*"  \
    | while IFS= read -r cmake_file; do
        proj_dir="$(dirname "$cmake_file")"
        if is_excluded "$proj_dir"; then
            echo "==> Skipping (excluded): $proj_dir"
            continue
        fi
        echo "==> Processing: $proj_dir"

        # 进入项目目录
        (
          cd "$proj_dir"

          # 创建和进入 build 目录
          mkdir -p build
          cd build

          # 配置与编译
          cmake .. || { echo "CMake configure failed in $proj_dir"; exit 1; }
          make -j4 || { echo "Make failed in $proj_dir"; exit 1; }
        )
      done

echo "All done."

popd