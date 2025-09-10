def payoff_call(S, K):
    """Calculate the payoff of a European call option.

    Parameters
    ----------
    S : float
        The spot price of the underlying asset at maturity.
    K : float
        The strike price of the option.

    Returns
    -------
    float
        The payoff of the call option.
    """
    return max(S - K, 0)

def payoff_put(S, K):
    """Calculate the payoff of a European put option.

    Parameters
    ----------
    S : float
        The spot price of the underlying asset at maturity.
    K : float
        The strike price of the option.

    Returns
    -------
    float
        The payoff of the put option.
    """
    return max(K - S, 0)
def payoff_forward(S, K):
    """Calculate the payoff of a forward contract.

    Parameters
    ----------
    S : float
        The spot price of the underlying asset at maturity.
    K : float
        The delivery price of the forward contract.

    Returns
    -------
    float
        The payoff of the forward contract.
    """
    return S - K
def payoff_futures(S, K):
    """Calculate the payoff of a futures contract.

    Parameters
    ----------
    S : float
        The spot price of the underlying asset at maturity.
    K : float
        The delivery price of the futures contract.

    Returns
    -------
    float
        The payoff of the futures contract.
    """
    return S - K
def payoff_digital_call(S, K):
    """Calculate the payoff of a digital call option.

    Parameters
    ----------
    S : float
        The spot price of the underlying asset at maturity.
    K : float
        The strike price of the option.

    Returns
    -------
    float
        The payoff of the digital call option.
    """
    return 1.0 if S > K else 0.0
def payoff_digital_put(S, K):
    """Calculate the payoff of a digital put option.

    Parameters
    ----------
    S : float
        The spot price of the underlying asset at maturity.
    K : float
        The strike price of the option.

    Returns
    -------
    float
        The payoff of the digital put option.
    """
    return 1.0 if S < K else 0.0
def payoff_straddle(S, K):
    """Calculate the payoff of a straddle option.

    Parameters
    ----------
    S : float
        The spot price of the underlying asset at maturity.
    K : float
        The strike price of the options.

    Returns
    -------
    float
        The payoff of the straddle option.
    """
    return abs(S - K)
def payoff_strangle(S, K1, K2):
    """Calculate the payoff of a strangle option.

    Parameters
    ----------
    S : float
        The spot price of the underlying asset at maturity.
    K1 : float
        The lower strike price of the options.
    K2 : float
        The upper strike price of the options.

    Returns
    -------
    float
        The payoff of the strangle option.
    """
    if S < K1:
        return K1 - S
    elif S > K2:
        return S - K2
    else:
        return 0.0
def payoff_butterfly(S, K1, K2, K3):
    """Calculate the payoff of a butterfly spread option.

    Parameters
    ----------
    S : float
        The spot price of the underlying asset at maturity.
    K1 : float
        The lower strike price of the options.
    K2 : float
        The middle strike price of the options.
        For symmetric butterfly, K2 = (K1 + K3) / 2
    K3 : float
        The upper strike price of the options.

    Returns
    -------
    float
        The payoff of the butterfly spread option.
    """
    if S < K1 or S > K3:
        return 0.0
    elif K1 <= S < K2:
        return S - K1
    elif K2 <= S <= K3:
        return K3 - S
    else:
        return 0.0
def payoff_condor(S, K1, K2, K3, K4):
    """Calculate the payoff of a condor spread option.

    Parameters
    ----------
    S : float
        The spot price of the underlying asset at maturity.
    K1 : float
        The lowest strike price of the options.
    K2 : float
        The lower-middle strike price of the options.
    K3 : float
        The upper-middle strike price of the options.
    K4 : float
        The highest strike price of the options.

    Returns
    -------
    float
        The payoff of the condor spread option.
    """
    if S < K1 or S > K4:
        return 0.0
    elif K1 <= S < K2:
        return S - K1
    elif K2 <= S < K3:
        return K2 - K1
    elif K3 <= S <= K4:
        return K4 - S
    else:
        return 0.0