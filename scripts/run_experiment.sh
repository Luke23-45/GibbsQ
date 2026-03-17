#!/usr/bin/env bash

# Execution script for GibbsQ experiments (Linux/macOS)
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
    echo "  policy   - Run policy comparison (Softmax vs JSQ, etc.) [DEPRECATED -> corrected_policy]"
    echo "  stress   - Run stress test (massive-N, critical load)"
    echo "  train    - [DEPRECATED] Run DGA routing agent training"
    echo "  fidelity - Run gradient survival check across horizons"
    echo "  n_train  - [DEPRECATED] Run neural curriculum training"
    echo "  parity   - [DEPRECATED] Run neural vs GibbsQ parity evaluation"
    echo "  jacobian - [DEPRECATED] Run Jacobian AD vs finite-difference check"
    echo "  stats    - Run 30-seed statistical significance benchmark"
    echo "  generalize - Run generalization stress heatmap"
    echo "  ablation - Run component ablation study"
    echo "  critical - Run critical stability boundary test"
    echo "  reinforce_train - Run REINFORCE SSA training (Track 1)"
    echo "  dr_train       - Run Domain Randomization training (Track 3)"
    echo "  corrected_policy - Run corrected Tiered benchmark (Track 4)"
    echo "  reinforce_check  - Run gradient validation (Track 5)"
    echo ""
    echo "Examples:"
    echo "  ./run_experiment.sh drift"
    echo "  ./run_experiment.sh sweep system.num_servers=5 simulation.ssa.sim_time=5000"
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
        PYTHON_SCRIPT="experiments.verification.drift_verification"
        ;;
    "sweep")
        PYTHON_SCRIPT="experiments.sweeps.stability_sweep"
        ;;
    "policy")
        PYTHON_SCRIPT="experiments.evaluation.policy_comparison"
        ;;
    "stress")
        PYTHON_SCRIPT="experiments.testing.stress_test"
        ;;
    "train")
        PYTHON_SCRIPT="experiments.training.train_dga"
        ;;
    "fidelity")
        PYTHON_SCRIPT="experiments.n_gibbsq.grad_check"
        ;;
    "n_train")
        PYTHON_SCRIPT="experiments.n_gibbsq.train"
        ;;
    "parity")
        PYTHON_SCRIPT="experiments.n_gibbsq.eval"
        ;;
    "jacobian")
        PYTHON_SCRIPT="experiments.n_gibbsq.jacobian_check"
        ;;
    "stats")
        PYTHON_SCRIPT="experiments.n_gibbsq.stats_bench"
        ;;
    "generalize")
        PYTHON_SCRIPT="experiments.n_gibbsq.gen_sweep"
        ;;
    "ablation")
        PYTHON_SCRIPT="experiments.n_gibbsq.ablation"
        ;;
    "critical")
        PYTHON_SCRIPT="experiments.n_gibbsq.critical_load"
        ;;
    "bias")
        PYTHON_SCRIPT="experiments.testing.verify_bias"
        ;;
    "reinforce_train")
        PYTHON_SCRIPT="experiments.n_gibbsq.train_reinforce"
        ;;
    "dr_train")
        PYTHON_SCRIPT="experiments.n_gibbsq.train_domain_randomized"
        ;;
    "corrected_policy")
        PYTHON_SCRIPT="experiments.evaluation.corrected_policy_comparison"
        ;;
    "reinforce_check")
        PYTHON_SCRIPT="experiments.testing.reinforce_gradient_check"
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
