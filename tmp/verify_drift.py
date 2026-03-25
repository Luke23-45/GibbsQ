import numpy as np
import math

def check_violation():
    # High heterogeneity case
    mu = np.array([10.0, 1.0])
    lam = 5.0
    alpha = 1.0  # High alpha to make it clear
    N = 2
    
    # State where Q1=10 (fast) but Q2=0 (slow)
    Q = np.array([10.0, 0.0])
    
    # Sojourn-time Softmax
    s = (Q + 1.0) / mu
    logits = -alpha * s
    logits -= logits.max()
    w = np.exp(logits)
    p = w / w.sum()
    
    # Bound from proof: sum p_i Q_i <= min Q_i + log N / alpha
    lhs = np.dot(p, Q)
    rhs = Q.min() + math.log(N) / alpha
    
    print(f"Policy: {p}")
    print(f"LHS (sum p_i Q_i): {lhs}")
    print(f"RHS (min Q_i + log N / alpha): {rhs}")
    
    if lhs > rhs + 1e-12:
        print("VIOLATION CONFIRMED")
    else:
        print("No violation at this state.")

if __name__ == "__main__":
    check_violation()
