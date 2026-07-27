#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
MODEL_DIR=$DIR/../models



# =================== Build all demo code =====================
EXCLUDE_DIRS=(
    "./legacy/Qwen2/demo_parallel"  # Qwen2's demo_parallel needs to be built separately
    "./legacy/Qwen1_5/demo_parallel"  # Qwen1_5's demo_parallel needs to be built separately
    "./legacy/Qwen/demo_parallel" # Qwen's demo_parallel needs to be built separately
    "./legacy/LWM/demo"
    "./legacy/VILA1_5/cpp_demo"
)

is_excluded() {
  local dir="$1"
  for ex in "${EXCLUDE_DIRS[@]}"; do
    # Normalize to a path relative to MODEL_DIR (starting with ./)
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

        # Enter the project directory
        (
          cd "$proj_dir"

          # Create and enter the build directory
          mkdir -p build
          cd build

          # Configure and build
          cmake .. || { echo "CMake configure failed in $proj_dir"; exit 1; }
          make -j4 || { echo "Make failed in $proj_dir"; exit 1; }
        )
      done

echo "All done."

popd