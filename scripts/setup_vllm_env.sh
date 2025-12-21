#!/bin/bash
# Setup isolated vLLM environment for GRAIL evaluation
# This creates a separate Python environment to avoid dependency conflicts with bittensor

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAIL_ROOT="$(dirname "$SCRIPT_DIR")"
VLLM_DIR="$GRAIL_ROOT/tools/vllm-server"

echo "========================================"
echo "GRAIL vLLM Environment Setup"
echo "========================================"
echo ""
echo "This creates an isolated vLLM environment at:"
echo "  $VLLM_DIR/.venv/"
echo ""

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not installed. Install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. vLLM requires CUDA GPUs."
fi

cd "$VLLM_DIR"

echo "Creating isolated Python 3.10 environment..."
uv venv --python 3.10

echo ""
echo "Installing vLLM and dependencies..."
echo "This may take several minutes on first run..."
echo ""

# Activate and install
source .venv/bin/activate
uv sync

# Verify installation
echo ""
echo "========================================"
echo "Verifying installation..."
echo "========================================"

VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "FAILED")
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "FAILED")
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "FAILED")

echo "vLLM version: $VLLM_VERSION"
echo "PyTorch version: $TORCH_VERSION"
echo "CUDA available: $CUDA_AVAILABLE"

if [ "$VLLM_VERSION" = "FAILED" ]; then
    echo ""
    echo "ERROR: vLLM installation failed!"
    exit 1
fi

deactivate

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "vLLM environment created at:"
echo "  $VLLM_DIR/.venv/"
echo ""
echo "To use vLLM backend, set:"
echo "  export GRAIL_VLLM_PYTHON=$VLLM_DIR/.venv/bin/python"
echo ""
echo "Or add to your .env file:"
echo "  GRAIL_VLLM_PYTHON=$VLLM_DIR/.venv/bin/python"
echo ""
echo "Verify manually with:"
echo "  $VLLM_DIR/.venv/bin/python -c \"import vllm; print(vllm.__version__)\""
echo ""
