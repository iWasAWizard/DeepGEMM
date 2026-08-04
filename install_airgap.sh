#!/usr/bin/env bash
# install_airgap.sh
#
# Run this on the airgapped aarch64 DGX after transferring deepgemm_bundle.tar.gz.
#
#   tar -xzf deepgemm_bundle.tar.gz
#   cd deepgemm_bundle
#   chmod +x cuda_compat.sh install_airgap.sh
#   ./cuda_compat.sh
#   source ~/.bashrc
#   ./install_airgap.sh
#
# Activate your venv before running if using one.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEELS_DIR="${SCRIPT_DIR}/wheels"
REPO_DIR="${SCRIPT_DIR}/repo"

die() { echo "ERROR: $*" >&2; exit 1; }
section() { echo ""; echo "================================================================"; echo "  $*"; echo "================================================================"; }

section "Preflight Checks"

ARCH=$(uname -m)
if [[ "${ARCH}" != "aarch64" ]]; then
    echo "WARNING: arch is '${ARCH}', expected 'aarch64'. Proceeding anyway."
fi

if ! command -v nvcc &>/dev/null; then
    die "nvcc not found in PATH. Try: export PATH=/usr/local/cuda/bin:\$PATH"
fi
echo "  nvcc:    $(nvcc --version | grep release | awk '{print $6}' | tr -d ',')"

if ! command -v g++ &>/dev/null; then
    die "g++ not found. Install with: apt install g++"
fi
echo "  g++:     $(g++ --version | head -1)"

PYTHON=$(command -v python3 || command -v python)
echo "  python:  $("${PYTHON}" --version 2>&1)"

[ -d "${WHEELS_DIR}" ] || die "wheels/ directory not found at ${WHEELS_DIR}"
echo "  wheels:  $(ls "${WHEELS_DIR}" | wc -l) files"

[ -d "${REPO_DIR}" ] || die "repo/ directory not found at ${REPO_DIR}"
[ -f "${REPO_DIR}/setup.py" ] || die "setup.py not found in repo/. Bundle may be corrupt."
echo "  repo:    OK"

section "Step 1/3 — Installing PyTorch from bundled wheels"

"${PYTHON}" -m pip install \
    --no-index \
    --find-links "${WHEELS_DIR}" \
    --no-deps \
    --break-system-packages \
    --upgrade \
    torch \
    torchvision \
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
    nvidia-nvjitlink-cu12 \
    typing_extensions \
    filelock \
    jinja2 \
    markupsafe \
    networkx \
    sympy \
    mpmath \
    fsspec \
    setuptools \
    wheel \
    packaging \
    numpy \
    pillow \
    2>&1 | grep -v "^Requirement already" || true

"${PYTHON}" -c "
import torch
print(f'  torch version:  {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:            {torch.cuda.get_device_name(0)}')
    cc = torch.cuda.get_device_capability(0)
    sm = cc[0] * 10 + cc[1]
    print(f'  SM{sm} detected')
    if sm < 90:
        print(f'  WARNING: DeepGEMM requires SM90 or SM100')
" || die "PyTorch verification failed. Check output above."

section "Step 2/3 — Building DeepGEMM from source"

cd "${REPO_DIR}"
export DG_FORCE_BUILD=1

[ -f "third-party/cutlass/CMakeLists.txt" ] || die "CUTLASS submodule missing."
[ -f "third-party/fmt/CMakeLists.txt" ] || die "fmt submodule missing."
echo "  Submodules: OK"

echo "  Running install.sh (DG_FORCE_BUILD=1)..."
bash install.sh 2>&1 | tee /tmp/deepgemm_install.log
INSTALL_EXIT=${PIPESTATUS[0]}

if [ ${INSTALL_EXIT} -ne 0 ]; then
    echo ""
    echo "install.sh failed. Last 30 lines:"
    tail -30 /tmp/deepgemm_install.log
    echo ""
    echo "Trying fallback: direct pip install..."
    "${PYTHON}" -m pip install \
        --no-build-isolation \
        --no-index \
        --find-links "${WHEELS_DIR}" \
        --break-system-packages \
        -e . 2>&1 | tee /tmp/deepgemm_pip.log \
        || die "Both install paths failed. See logs above."
fi

section "Step 3/3 — Smoke Test"

"${PYTHON}" -c "
import deep_gemm
import torch
print('  deep_gemm imported successfully')
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    sm = cc[0] * 10 + cc[1]
    if sm >= 100:
        print(f'  SM{sm} (Blackwell) — full NT/TN/NN/TT layout support')
    elif sm >= 90:
        print(f'  SM{sm} (Hopper) — NT layout support')
    else:
        print(f'  WARNING: SM{sm} not supported (need SM90+)')
" || echo "WARNING: Smoke test failed. Check logs above."

section "Installation Complete"
cat << 'EOF'
  DeepGEMM installed in development mode from the bundled repo.

  First import triggers JIT compilation — slow once, cached after.
  Cache location: ~/.deep_gemm (override with DG_JIT_CACHE_DIR)

  Quick verification:
    python3 -c "import deep_gemm; print('OK')"

  Full test suite:
    cd deepgemm_bundle/repo
    python3 tests/test_layout.py
    python3 tests/test_core.py

  Useful env vars:
    DG_JIT_DEBUG=1          verbose JIT output
    DG_PRINT_CONFIGS=1      show selected kernel configs per shape
    DG_JIT_CACHE_DIR=/path  override kernel cache location
    DG_JIT_USE_NVRTC=1      faster compilation (slight perf tradeoff)
EOF
