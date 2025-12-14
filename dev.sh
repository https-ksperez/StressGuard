#!/bin/bash
# StressGuard Development Helper Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Display usage
usage() {
    cat << EOF
StressGuard Development Helper

Usage: ./dev.sh [command]

Commands:
    install         Install dependencies
    install-dev     Install development dependencies
    test            Run all tests
    test-ml         Run ML toolkit tests
    test-dl         Run DL toolkit tests
    test-utils      Run utility tests
    example-ml      Run ML example
    example-dl      Run DL example
    example-bot     Run chatbot example
    run             Run interactive chatbot
    clean           Clean temporary files and caches
    help            Show this help message

Examples:
    ./dev.sh install
    ./dev.sh test
    ./dev.sh run
EOF
}

# Install dependencies
install() {
    info "Installing dependencies..."
    pip install -r requirements.txt
}

# Install development dependencies
install_dev() {
    info "Installing development dependencies..."
    pip install -r requirements.txt
    pip install pytest pytest-cov black flake8
}

# Run tests
run_tests() {
    info "Running all tests..."
    PYTHONPATH=src pytest tests/
}

# Run ML tests
run_ml_tests() {
    info "Running ML toolkit tests..."
    PYTHONPATH=src pytest tests/test_ml_toolkit.py -v
}

# Run DL tests
run_dl_tests() {
    info "Running DL toolkit tests..."
    PYTHONPATH=src pytest tests/test_dl_toolkit.py -v
}

# Run utils tests
run_utils_tests() {
    info "Running utility tests..."
    PYTHONPATH=src pytest tests/test_utils.py -v
}

# Run ML example
run_ml_example() {
    info "Running ML example..."
    PYTHONPATH=src python examples/ml_example.py
}

# Run DL example
run_dl_example() {
    info "Running DL example..."
    PYTHONPATH=src python examples/dl_example.py
}

# Run chatbot example
run_bot_example() {
    info "Running chatbot example..."
    warning "Note: This requires downloading the LLM model (~500MB)"
    PYTHONPATH=src python examples/basic_chatbot.py
}

# Run interactive chatbot
run_chatbot() {
    info "Starting StressGuard chatbot..."
    warning "Note: This requires downloading the LLM model (~500MB)"
    PYTHONPATH=src python src/chatbot/main.py
}

# Clean temporary files
clean() {
    info "Cleaning temporary files..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
    info "Cleanup complete!"
}

# Main script logic
case "${1:-}" in
    install)
        install
        ;;
    install-dev)
        install_dev
        ;;
    test)
        run_tests
        ;;
    test-ml)
        run_ml_tests
        ;;
    test-dl)
        run_dl_tests
        ;;
    test-utils)
        run_utils_tests
        ;;
    example-ml)
        run_ml_example
        ;;
    example-dl)
        run_dl_example
        ;;
    example-bot)
        run_bot_example
        ;;
    run)
        run_chatbot
        ;;
    clean)
        clean
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        error "Unknown command: ${1:-}"
        echo ""
        usage
        exit 1
        ;;
esac
