import numpy as np

def payoff_call(S, K):
    """European call: max(S - K, 0). Works with arrays (broadcasts)."""
    return np.maximum(S - K, 0.0)

def payoff_put(S, K):
    """European put: max(K - S, 0). Works with arrays (broadcasts)."""
    return np.maximum(K - S, 0.0)

def payoff_forward(S, K):
    """Forward payoff: S - K. Works with arrays (broadcasts)."""
    return np.asarray(S) - K

def payoff_futures(S, K):
    """Futures payoff: S - K (same as forward at expiry)."""
    return np.asarray(S) - K

def payoff_digital_call(S, K):
    """Digital (cash-or-nothing) call paying 1 if S > K else 0."""
    return (np.asarray(S) > K).astype(float)

def payoff_digital_put(S, K):
    """Digital (cash-or-nothing) put paying 1 if S < K else 0."""
    return (np.asarray(S) < K).astype(float)

def payoff_straddle(S, K):
    """Straddle: |S - K| = call(K) + put(K)."""
    return np.abs(np.asarray(S) - K)

def payoff_strangle(S, K1, K2):
    """
    Strangle (K1 < K2): max(K1 - S, 0) + max(S - K2, 0).
    Vectorized piecewise version via np.where.
    """
    S = np.asarray(S)
    return np.where(S < K1, K1 - S, np.where(S > K2, S - K2, 0.0))

def payoff_butterfly(S, K1, K2, K3):
    """
    Long call butterfly with strikes K1 < K2 < K3.
    Piecewise triangular payoff in [K1, K3], peak at K2.
    Equivalent to: call(K1) - 2*call(K2) + call(K3).
    """
    S = np.asarray(S)
    return np.where((S < K1) | (S > K3), 0.0,
                    np.where(S < K2, S - K1, K3 - S))

def payoff_condor(S, K1, K2, K3, K4):
    """
    Long call condor with K1 < K2 < K3 < K4.
    Plateau (K2-K1) between K2 and K3.
    Equivalent to: call(K1) - call(K2) - call(K3) + call(K4).
    """
    S = np.asarray(S)
    return np.where((S < K1) | (S > K4), 0.0,
                    np.where(S < K2, S - K1,
                    np.where(S < K3, K2 - K1, K4 - S)))
