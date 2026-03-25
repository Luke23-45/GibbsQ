import numpy as np
import math

def softmax(x, alpha):
    logits = -alpha * x
    logits -= np.max(logits)
    w = np.exp(logits)
    return w / np.sum(w)

def check_violation(N, lam, mu, alpha, Q, mode="sojourn"):
    mu = np.array(mu)
    Q = np.array(Q)
    
    # Exact Drift
    if mode == "sojourn":
        p = softmax((Q + 1.0) / mu, alpha)
    else:
        p = softmax(Q, alpha)
        
    arrival_term = lam * np.dot(p, Q)
    service_term = np.dot(mu, Q)
    active = (Q > 0).astype(float)
    C_Q = lam / 2.0 + 0.5 * np.dot(mu, active)
    drift = arrival_term - service_term + C_Q
    
    # Proof Bound
    cap = np.sum(mu)
    Q_min = np.min(Q)
    delta = Q - Q_min
    R = (lam * math.log(N)) / alpha + (lam + cap) / 2.0
    bound = -(cap - lam) * Q_min - np.dot(mu, delta) + R
    
    return drift, bound, drift > bound + 1e-12

# Heterogeneous case from drift.yaml
N = 2
lam = 2.375
mu = [1.0, 1.5]
alpha = 1.0

# Check a state where the mismatch might show up
# e.g. large Q on fast server, small Q on slow server
Q = [2, 20] # Q_min = 2, delta = [0, 18]
# Sojourn: (2+1)/1 = 3, (20+1)/1.5 = 14
# Raw: 2, 20

d_raw, b_raw, v_raw = check_violation(N, lam, mu, alpha, Q, mode="raw")
d_soj, b_soj, v_soj = check_violation(N, lam, mu, alpha, Q, mode="sojourn")

print(f"State Q={Q}, lam={lam}, mu={mu}, alpha={alpha}")
print(f"RAW Mode:     Drift={d_raw:.4f}, Bound={b_raw:.4f}, Violated={v_raw}")
print(f"SOJOURN Mode: Drift={d_soj:.4f}, Bound={b_soj:.4f}, Violated={v_soj}")

# Try to find a violation for Sojourn
for q1 in range(50):
    for q2 in range(50):
        Q_test = [q1, q2]
        _, _, v = check_violation(N, lam, mu, alpha, Q_test, mode="sojourn")
        if v:
            print(f"VIOLATION FOUND AT Q={Q_test}")
            break
