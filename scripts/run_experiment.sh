#!/usr/bin/env bash

# Robust execution script for MoEQ experiments (Linux/macOS)
# Usage: ./run_experiment.sh <experiment_name> [hydra_overrides...]

set -euo pipefail

# Determine the absolute path of the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set PYTHONPATH so Python can find the `src` module
export PYTHONPATH="$PROJECT_ROOT"

print_usage() {
    echo "Usage: ./run_experiment.sh <experiment> [hydra_args...]"
    echo ""
    echo "Available experiments:"
    echo "  drift    - Run drift verification (drift_vs_norm, heatmap)"
    echo "  sweep    - Run stability sweep across alpha and rho"
    echo "  policy   - Run policy comparison (Softmax vs JSQ, etc.)"
    echo ""
    echo "Examples:"
    echo "  ./run_experiment.sh drift"
    echo "  ./run_experiment.sh sweep system.num_servers=5 simulation.sim_time=5000"
    echo "  ./run_experiment.sh policy +simulation.export_trajectories=True"
    echo ""
}

if [ "$#" -lt 1 ]; then
    echo "Error: Missing experiment name."
    print_usage
    exit 1
fi

EXPERIMENT=$1
shift # The rest of the arguments are passed to Python/Hydra

PYTHON_SCRIPT=""
case "$EXPERIMENT" in
    "drift")
        PYTHON_SCRIPT="experiments.drift_verification"
        ;;
    "sweep")
        PYTHON_SCRIPT="experiments.stability_sweep"
        ;;
    "policy")
        PYTHON_SCRIPT="experiments.policy_comparison"
        ;;
    "-h"|"--help"|"help")
        print_usage
        exit 0
        ;;
    *)
        echo "Error: Unknown experiment '$EXPERIMENT'"
        print_usage
        exit 1
        ;;
esac

echo "=========================================================="
echo " Starting Experiment: $EXPERIMENT"
echo " Remaining Args (Hydra Overrides): $*"
echo "=========================================================="

cd "$PROJECT_ROOT"
python -m "$PYTHON_SCRIPT" "$@"
