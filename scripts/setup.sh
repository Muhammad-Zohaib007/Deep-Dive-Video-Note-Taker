#!/usr/bin/env bash
# =============================================================================
# Deep-Dive Video Note Taker — Bare-Metal Setup Script
#
# Installs all dependencies for running on a local machine (no Docker).
# Supports Linux and macOS. For Windows, use WSL or see README.
#
# Usage:
#   chmod +x scripts/setup.sh
#   ./scripts/setup.sh
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No color

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ---- System Detection -------------------------------------------------------

OS="$(uname -s)"
ARCH="$(uname -m)"
info "Detected: ${OS} ${ARCH}"

# ---- Check Python ------------------------------------------------------------

info "Checking Python..."
if command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    error "Python 3.10+ is required but not found."
    error "Install it from: https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]; }; then
    error "Python 3.10+ required. Found: ${PYTHON_VERSION}"
    exit 1
fi
ok "Python ${PYTHON_VERSION} found."

# ---- Check FFmpeg ------------------------------------------------------------

info "Checking FFmpeg..."
if command -v ffmpeg &>/dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -n1)
    ok "FFmpeg found: ${FFMPEG_VERSION}"
else
    warn "FFmpeg not found. Installing..."
    if [ "$OS" = "Darwin" ]; then
        if command -v brew &>/dev/null; then
            brew install ffmpeg
        else
            error "Homebrew not found. Install FFmpeg manually: https://ffmpeg.org"
            exit 1
        fi
    elif [ "$OS" = "Linux" ]; then
        if command -v apt-get &>/dev/null; then
            sudo apt-get update && sudo apt-get install -y ffmpeg
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y ffmpeg
        elif command -v pacman &>/dev/null; then
            sudo pacman -S --noconfirm ffmpeg
        else
            error "Could not determine package manager. Install FFmpeg manually."
            exit 1
        fi
    fi
    ok "FFmpeg installed."
fi

# ---- Check / Install Poetry -------------------------------------------------

info "Checking Poetry..."
if command -v poetry &>/dev/null; then
    POETRY_VERSION=$(poetry --version 2>&1)
    ok "Poetry found: ${POETRY_VERSION}"
else
    warn "Poetry not found. Installing..."
    $PYTHON -m pip install --user poetry
    # Refresh PATH
    export PATH="$HOME/.local/bin:$PATH"
    if command -v poetry &>/dev/null; then
        ok "Poetry installed."
    else
        error "Poetry installed but not in PATH. Add ~/.local/bin to PATH."
        exit 1
    fi
fi

# ---- Install Project Dependencies -------------------------------------------

info "Installing project dependencies with Poetry..."
poetry install

ok "Python dependencies installed."

# ---- Check / Install Ollama -------------------------------------------------

info "Checking Ollama..."
if command -v ollama &>/dev/null; then
    ok "Ollama found."
else
    warn "Ollama not found. Installing..."
    if [ "$OS" = "Darwin" ]; then
        if command -v brew &>/dev/null; then
            brew install ollama
        else
            warn "Download Ollama from: https://ollama.com/download"
        fi
    elif [ "$OS" = "Linux" ]; then
        curl -fsSL https://ollama.com/install.sh | sh
    fi

    if command -v ollama &>/dev/null; then
        ok "Ollama installed."
    else
        warn "Ollama not installed automatically. Download from: https://ollama.com/download"
    fi
fi

# ---- Pull Default LLM Model -------------------------------------------------

info "Checking Ollama model (llama3.1:8b)..."
if command -v ollama &>/dev/null; then
    # Start Ollama in background if not running
    if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
        info "Starting Ollama server..."
        ollama serve &>/dev/null &
        sleep 3
    fi

    # Check if model exists
    if ollama list 2>/dev/null | grep -q "llama3.1:8b"; then
        ok "Model llama3.1:8b already available."
    else
        info "Pulling llama3.1:8b model (this may take a while)..."
        ollama pull llama3.1:8b
        ok "Model llama3.1:8b pulled."
    fi
else
    warn "Ollama not available. Skipping model pull."
    warn "Install Ollama and run: ollama pull llama3.1:8b"
fi

# ---- Create User Config Directory -------------------------------------------

info "Setting up user config directory..."
CONFIG_DIR="$HOME/.notetaker"
mkdir -p "$CONFIG_DIR/videos" "$CONFIG_DIR/models" "$CONFIG_DIR/chroma" "$CONFIG_DIR/logs"

if [ ! -f "$CONFIG_DIR/config.yaml" ]; then
    if [ -f "config.default.yaml" ]; then
        cp config.default.yaml "$CONFIG_DIR/config.yaml"
        ok "Default config copied to ${CONFIG_DIR}/config.yaml"
    fi
else
    ok "User config already exists at ${CONFIG_DIR}/config.yaml"
fi

# ---- Summary -----------------------------------------------------------------

echo ""
echo "============================================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "============================================================"
echo ""
echo "Quick start:"
echo "  1. Start Ollama:    ollama serve"
echo "  2. Process a video: poetry run notetaker process <youtube_url>"
echo "  3. Start web UI:    poetry run notetaker serve"
echo "  4. Open browser:    http://localhost:8000"
echo ""
echo "Run tests:"
echo "  poetry run pytest"
echo ""
echo "Docker alternative:"
echo "  docker-compose up"
echo ""
