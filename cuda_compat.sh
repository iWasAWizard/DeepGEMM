#!/usr/bin/env bash
# cuda_compat.sh
#
# Creates CUDA 12 soname symlinks pointing at CUDA 13 libraries.
# Run once on the DGX before running install_airgap.sh.
#
#   chmod +x cuda_compat.sh
#   ./cuda_compat.sh
#   source ~/.bashrc

set -euo pipefail

CUDA_LIB="/usr/local/cuda-13.0/targets/sbsa-linux/lib"
COMPAT_DIR="${HOME}/cuda-compat"

if [ ! -d "${CUDA_LIB}" ]; then
    echo "ERROR: ${CUDA_LIB} not found. Locate your CUDA lib dir with:"
    echo "  find /usr/local -name 'libcudart.so*' 2>/dev/null"
    exit 1
fi

mkdir -p "${COMPAT_DIR}"
echo "Creating CUDA 12 compat symlinks in ${COMPAT_DIR}..."
echo ""

LIBS=(
    libcudart
    libcublas
    libcublasLt
    libcufft
    libcurand
    libcusolver
    libcusparse
    libnvrtc
    libnvToolsExt
    libcufile
    libcuda
)

for lib in "${LIBS[@]}"; do
    src=$(find "${CUDA_LIB}" -name "${lib}.so.1*" 2>/dev/null | sort | tail -1)
    if [ -n "$src" ]; then
        ln -sf "$src" "${COMPAT_DIR}/${lib}.so.12"
        echo "  linked: ${lib}.so.12 -> $(basename $src)"
    else
        echo "  skipped: ${lib} not found in ${CUDA_LIB}"
    fi
done

echo ""
echo "Adding LD_LIBRARY_PATH to ~/.bashrc..."

EXPORT_LINE="export LD_LIBRARY_PATH=${COMPAT_DIR}:${CUDA_LIB}:\${LD_LIBRARY_PATH:-}"

if grep -qF "${COMPAT_DIR}" ~/.bashrc 2>/dev/null; then
    echo "  Already present in ~/.bashrc, skipping."
else
    echo "" >> ~/.bashrc
    echo "# CUDA 12/13 compat for PyTorch" >> ~/.bashrc
    echo "${EXPORT_LINE}" >> ~/.bashrc
    echo "  Added to ~/.bashrc"
fi

echo ""
echo "Applying to current shell..."
export LD_LIBRARY_PATH="${COMPAT_DIR}:${CUDA_LIB}:${LD_LIBRARY_PATH:-}"

echo ""
echo "================================================================"
echo "  Done. LD_LIBRARY_PATH is set for this session."
echo "  For future sessions it will be loaded from ~/.bashrc."
echo ""
echo "  Next step:"
echo "    ./install_airgap.sh"
echo "================================================================"
