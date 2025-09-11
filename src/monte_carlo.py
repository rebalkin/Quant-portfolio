import numpy as np

# Single-step geometric Brownian motion

def GBM_single(S0, mu, sigma, dt):
    """
    Simulate a single step of geometric Brownian motion.

    Parameters:
        S0 (float): Initial asset price
        mu (float): Expected return (drift)
        sigma (float): Volatility
        dt (float): Time increment

    Returns:
        float: Simulated asset price after one time step
    """
    Z = np.random.normal()
    S1 = S0 * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return S1

def GBM_time_series(S0, mu, sigma, T,dt):
    n = T/dt
    prices = [S0]
    for i in range(n):
        t = i*dt
        drift = mu(t) if callable(mu) else mu
        vol = sigma(t) if callable(sigma) else sigma
        S_init = prices[-1]
        S_next = GBM_single(S_init, drift, vol, dt)
        prices.append(S_next)
    return np.array(prices)
# TO DO - correct error estimation
def MC_price(S0, r, sigma, T,dt,payoff,N):
    if not callable(payoff):
        print("Error: payoff needs to ba callable")
        return None
    if not callable(r) and not callable(sigma):
        return MC_price_consts(S0, r, sigma, T,payoff,N)
    res=0
    for i in range(N):
        S = GBM_time_series(S0, r, sigma, T,dt)
        res += payoff(S[-1])
    average = res/N
    if not callable(r):
        result = np.exp(-r*T)*average
        error = result/np.sqrt(N)
        return result, error
    else:
        # r is a function of time
        integral_r = np.trapzoid([r(t) for t in np.arange(0,T+dt,dt)], dx=dt)
        result = np.exp(-integral_r)*average
        error = result/np.sqrt(N)
        return result, error
    
    
    

def MC_price_consts(S0, r, sigma, T,payoff,N):
    # This can be used the validate Black-Scholes 
    if not callable(payoff):
        print("Error: payoff needs to ba callable")
        return None
    res=[]
    for i in range(N):
        S = GBM_single(S0, r, sigma, T)
        res.append(np.exp(-r*T)*payoff(S))
    res = np.array(res)

    mean = res.mean()              # or np.mean(arr)
    std = res.std(ddof=1)    

    return mean, std/np.sqrt(N)
        


    