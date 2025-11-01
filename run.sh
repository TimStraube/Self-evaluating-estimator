#!/bin/bash

# ============================================================================
# Self-Evaluating Estimator (SEE) - Training Script
# ============================================================================
# This script runs the SEE reinforcement learning agent in the global 
# environment with configurable parameters.
# ============================================================================

set -e  # Exit on error

# Color codes for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============================================================================
# Configuration
# ============================================================================

# Default parameters
TIMESTEPS=${TIMESTEPS:-10000}
MEMORY_CAPACITY=${MEMORY_CAPACITY:-5}
VERBOSE=${VERBOSE:-1}
MODEL=${MODEL:-PPO}
POLICY=${POLICY:-MlpPolicy}

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo -e "${CYAN}============================================================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}============================================================================${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# Main Script
# ============================================================================

print_header "Self-Evaluating Estimator (SEE) - Agent Training"

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    print_error "Poetry is not installed. Please install it first:"
    echo "  curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

print_success "Poetry found"

# Check if virtual environment is set up
if [ ! -d ".venv" ]; then
    print_warning "Virtual environment not found. Setting up..."
    poetry install
    print_success "Virtual environment created"
else
    print_info "Virtual environment found"
fi

# Display configuration
print_header "Configuration"
echo -e "  ${CYAN}Model:${NC}            $MODEL"
echo -e "  ${CYAN}Policy:${NC}           $POLICY"
echo -e "  ${CYAN}Total Timesteps:${NC}  $TIMESTEPS"
echo -e "  ${CYAN}Memory Capacity:${NC}  $MEMORY_CAPACITY"
echo -e "  ${CYAN}Verbose Level:${NC}    $VERBOSE"
echo ""

# Display available environments
print_header "Available Environments"
print_info "Global environment includes:"
echo "  - Random Environment (random.py)"
echo "  - Test Environment (test.py)"
echo "  - Battleship Environment (battleship.py)"
echo ""

# Run the agent
print_header "Starting Agent Training"
print_info "Running SEE agent in global environment..."
echo ""

# Execute the training script using Poetry
poetry run python -m src.agents.main

# Check if training was successful
if [ $? -eq 0 ]; then
    print_header "Training Complete"
    print_success "Agent training completed successfully!"
    echo ""
    print_info "Next steps:"
    echo "  - View logs in test/logs/"
    echo "  - Run benchmarks with: bash benchmark.sh"
    echo "  - Test agent with: bash test.sh"
else
    print_error "Training failed. Check the output above for errors."
    exit 1
fi

print_header "Done"
