import numpy as np
from scipy.optimize import minimize


def solve_dual_discrete(target_q_t, action_p_t, last_eta, eps):
    target_q = target_q_t.cpu().transpose(0, 1).numpy()
    action_p = action_p_t.cpu().transpose(0, 1).numpy()
    if last_eta < 1e-6:
        last_eta = 1e-6

    def dual(eta):
        max_q = np.max(target_q, 1)
        return eta * eps + np.mean(max_q) \
            + eta * np.mean(np.log(np.sum(
                action_p * np.exp((target_q - max_q[:, None]) / eta), axis=1)))

    # Minimize dual function
    bounds = [(1e-6, None)]
    res = minimize(dual, np.array([last_eta]), method='SLSQP', bounds=bounds)
    res = res.x[0]
    return res


def solve_dual_continuous(target_q_t, last_eta, eps):
    target_q = target_q_t.cpu().transpose(0, 1).numpy()  # (K, N)
    if last_eta < 1e-6:
        last_eta = 1e-6

    def dual(eta):
        max_q = np.max(target_q, 1)
        return eta * eps + np.mean(max_q) \
            + eta * np.mean(np.log(np.mean(np.exp((target_q - max_q[:, None]) / eta), axis=1)))

    # Minimize dual function
    bounds = [(1e-6, None)]
    res = minimize(dual, np.array([last_eta]), method='SLSQP', bounds=bounds)
    res = res.x[0]
    return res
