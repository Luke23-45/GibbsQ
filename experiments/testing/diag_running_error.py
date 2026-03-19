"""Diagnostic 7: Running L2 Error Demonstration."""
import numpy as np

# Hypothesis: The running RelErr is just an aggregate L2 norm of independent variables.
# It "increases" only because the first few variables randomly happened to have lower error,
# or because L2 norm naturally accumulates the variance over more dimensions.

np.random.seed(42)

# Simulate 100 parameters with TRUE gradient values (Signal)
N = 100
true_grads = np.random.normal(0, 0.05, N)

# Simulate FD gradient estimates with independent noise (Signal + Noise)
# Our previous tests showed FD has about 15-20% relative error in L2 norm
noise = np.random.normal(0, 0.01, N)
fd_grads = true_grads + noise

print("Hypothesis Test: Tracking Running L2 Error over independent parameters")
print("-" * 75)

computed_mask = np.zeros(N, dtype=bool)

for i in range(N):
    computed_mask[i] = True
    
    # 1. Per-Parameter Relative Error
    rf_val = true_grads[i]
    fd_val = fd_grads[i]
    abs_diff = abs(rf_val - fd_val)
    param_rel_err = abs_diff / max(1e-12, abs(fd_val))
    
    # 2. Running Aggregate L2 Relative Error
    rf_sub = true_grads[computed_mask]
    fd_sub = fd_grads[computed_mask]
    
    diff_norm = np.linalg.norm(rf_sub - fd_sub)
    fd_norm_sub = np.linalg.norm(fd_sub)
    running_rel_err = diff_norm / fd_norm_sub
    
    if (i + 1) % 10 == 0:
        print(f"Param {(i+1):3d}/{N}: ParamErr = {param_rel_err:7.4f} | Running_L2AggErr = {running_rel_err:7.4f}")

print("\nConclusion: The 'Running_L2AggErr' converges to the global noise-to-signal ratio.")
print("It is NOT a timeline of a model training where error should decrease.")
