#!/usr/bin/env bash
# bundle_deepgemm.sh
#
# Run this on any internet-connected machine to collect everything needed to
# build and install DeepGEMM on an airgapped aarch64 DGX (SM100 / GB200).
#
# Output: deepgemm_bundle.tar.gz
#
# Usage:
#   chmod +x bundle_deepgemm.sh
#   PYTHON_VERSION=3.12.3 CUDA_VERSION=128 ./bundle_deepgemm.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_URL="${REPO_URL:-https://github.com/iWasAWizard/DeepGEMM.git}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12.3}"
CUDA_VERSION="${CUDA_VERSION:-128}"
BUNDLE_DIR="${BUNDLE_DIR:-deepgemm_bundle}"
OUTPUT_ARCHIVE="deepgemm_bundle.tar.gz"

# Strip to major.minor only (handles 3.12.3 -> cp312)
PY_SHORT="$(echo "${PYTHON_VERSION}" | cut -d. -f1,2)"
PY_TAG="cp$(echo "${PY_SHORT}" | tr -d '.')"

TORCH_INDEX_URL="https://download.pytorch.org/whl/cu${CUDA_VERSION}"

echo "================================================================"
echo "  DeepGEMM Airgap Bundle Script"
echo "  Target: linux/aarch64, Python ${PY_SHORT} (${PY_TAG}), CUDA ${CUDA_VERSION}"
echo "================================================================"
echo ""

# ---------------------------------------------------------------------------
# Clean slate
# ---------------------------------------------------------------------------
rm -rf "${BUNDLE_DIR}"
mkdir -p "${BUNDLE_DIR}/wheels"
mkdir -p "${BUNDLE_DIR}/repo"

# ---------------------------------------------------------------------------
# 1. Clone repo with all submodules (cutlass + fmt)
# ---------------------------------------------------------------------------
echo "[1/5] Cloning ${REPO_URL} with submodules..."
git clone --recursive --depth=1 "${REPO_URL}" "${BUNDLE_DIR}/repo"

if [ ! -f "${BUNDLE_DIR}/repo/third-party/cutlass/CMakeLists.txt" ]; then
    echo "ERROR: CUTLASS submodule did not clone."
    exit 1
fi
if [ ! -f "${BUNDLE_DIR}/repo/third-party/fmt/CMakeLists.txt" ]; then
    echo "ERROR: fmt submodule did not clone."
    exit 1
fi
echo "  -> Submodules OK (cutlass + fmt)"

# ---------------------------------------------------------------------------
# 2. Download PyTorch + torchvision for linux/aarch64
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Downloading PyTorch wheels (aarch64 / ${PY_TAG} / cu${CUDA_VERSION})..."

pip download \
    --dest "${BUNDLE_DIR}/wheels" \
    --platform "manylinux_2_28_aarch64" \
    --python-version "${PY_SHORT}" \
    --implementation "cp" \
    --abi "${PY_TAG}" \
    --only-binary ":all:" \
    --no-deps \
    --index-url "${TORCH_INDEX_URL}" \
    torch torchvision

# ---------------------------------------------------------------------------
# 3. Download nvidia CUDA packages for aarch64.
#    These are NOT system-provided on this DGX and must come from the bundle.
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Downloading nvidia CUDA packages (aarch64)..."

pip download \
    --dest "${BUNDLE_DIR}/wheels" \
    --platform "manylinux2014_aarch64" \
    --only-binary ":all:" \
    --no-deps \
    nvidia-cufft-cu12 \
    nvidia-curand-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-cuda-nvrtc-cu12 \
    nvidia-cuda-runtime-cu12 \
    nvidia-cuda-cupti-cu12 \
    nvidia-cublas-cu12 \
    nvidia-cudnn-cu12 \
    nvidia-cusparselt-cu12 \
    nvidia-nccl-cu12 \
    nvidia-nvtx-cu12 \
    nvidia-nvjitlink-cu12

pip download \
    --dest deepgemm_bundle/wheels \
    --platform "manylinux_2_18_aarch64" \
    --only-binary ":all:" \
    --no-deps \
    nvidia-nccl-cu12

# ---------------------------------------------------------------------------
# 4. Download pure-Python dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Downloading pure-Python dependencies..."

pip download \
    --dest "${BUNDLE_DIR}/wheels" \
    --only-binary ":all:" \
    --no-deps \
    typing_extensions \
    filelock \
    jinja2 \
    networkx \
    sympy \
    mpmath \
    fsspec \
    setuptools \
    wheel \
    packaging \
    numpy \
    pillow

# markupsafe has a compiled extension, needs platform flag
pip download \
    --dest "${BUNDLE_DIR}/wheels" \
    --platform "manylinux_2_28_aarch64" \
    --python-version "${PY_SHORT}" \
    --implementation "cp" \
    --abi "${PY_TAG}" \
    --only-binary ":all:" \
    --no-deps \
    markupsafe

# ---------------------------------------------------------------------------
# 5. Write bundle manifest and copy install script
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] Packing bundle..."

cat > "${BUNDLE_DIR}/BUNDLE_INFO.txt" << EOF
DeepGEMM Airgap Bundle
======================
Created:         $(date -u +"%Y-%m-%d %H:%M UTC")
Repo:            ${REPO_URL}
Repo commit:     $(git -C "${BUNDLE_DIR}/repo" rev-parse HEAD)
Target arch:     linux/aarch64 (DGX GB200 / Grace CPU)
Target SM:       SM100 (Blackwell)
Python:          ${PYTHON_VERSION} (${PY_TAG})
CUDA index:      cu${CUDA_VERSION}

Notes:
  - ninja intentionally excluded (system-provided on DGX)
  - JIT kernel compilation happens at first import, not install time
  - nvcc must be in PATH on the DGX (it will be on stock DGX OS)
  - Kernel cache: ~/.deep_gemm (override with DG_JIT_CACHE_DIR)
  - Run cuda_compat.sh before install_airgap.sh if not already done
EOF

cp "$(dirname "$0")/install_airgap.sh" "${BUNDLE_DIR}/" 2>/dev/null \
    || echo "  WARNING: install_airgap.sh not found alongside this script — copy it manually"
cp "$(dirname "$0")/cuda_compat.sh" "${BUNDLE_DIR}/" 2>/dev/null \
    || echo "  WARNING: cuda_compat.sh not found alongside this script — copy it manually"

tar -czf "${OUTPUT_ARCHIVE}" "${BUNDLE_DIR}/"
rm -rf "${BUNDLE_DIR}"

BUNDLE_SIZE=$(du -sh "${OUTPUT_ARCHIVE}" | cut -f1)
echo ""
echo "================================================================"
echo "  Bundle complete: ${OUTPUT_ARCHIVE}  (${BUNDLE_SIZE})"
echo ""
echo "  Transfer to the DGX:"
echo "    scp ${OUTPUT_ARCHIVE} user@dgx:/home/user/"
echo ""
echo "  Then on the DGX:"
echo "    tar -xzf ${OUTPUT_ARCHIVE}"
echo "    cd deepgemm_bundle"
echo "    chmod +x cuda_compat.sh install_airgap.sh"
echo "    ./cuda_compat.sh"
echo "    source ~/.bashrc"
echo "    ./install_airgap.sh"
echo "================================================================"
