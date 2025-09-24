import numpy as np
from scipy.stats import norm
from scipy import integrate
from scipy.optimize import brentq

# (norm.cdf)

def d1_func(S, E, r, D,sigma, T):
    return (np.log(S / E) + (r-D + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
def d2_func(S, E, r,D, sigma, T):
    return d1_func(S, E, r, D,sigma, T)-sigma * np.sqrt(T)
def N(x):
    return norm.cdf(x)
def Np(x):
    return norm.pdf(x)

def bs_formula_price(S, E, r, D,sigma, T, option_type):
    d1 = d1_func(S, E, r, D,sigma, T)
    d2 = d2_func(S, E, r,D, sigma, T)
    if option_type=="call":
        return S*np.exp(-D*T)*N(d1)-E*np.exp(-r*T)*N(d2)
    elif option_type=="put":
        return E*np.exp(-r*T)*N(-d2)-S*np.exp(-D*T)*N(-d1)
    elif option_type=="digital_call":
        return np.exp(-r*T)*N(d2)
    elif option_type=="digital_put":
        return np.exp(-r*T)*N(-d2)
    else:
        raise ValueError("option_type must be either 'call', 'put', 'digital_call' or 'digital_put'")

def ST_rand(S, r, D, sigma, T, y):
    return S * np.exp((r-D-0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * y)


def bs_integral_price(S, r, D, sigma, T,payoff):
    result, error = integrate.quad(lambda x: payoff(ST_rand(S, r, D, sigma, T,x)), -np.inf, np.inf)

def implied_vol(S, E, r, D, T, option_type,price):
    objective = lambda sigma: bs_formula_price(S, E, r, D,sigma, T, option_type) - price
    try:
        return brentq(objective, 1e-6, 2)  # search between 0.000001 and 500%
    except ValueError:
        return np.nan
    

def delta(S, E, r, D,sigma, T, option_type):
    d1 = d1_func(S, E, r, D,sigma, T)
    if option_type=="call":
        return np.exp(-D*T)*N(d1)
    elif option_type=="put":
        return np.exp(-D*T)*(N(d1)-1)
    elif option_type=="digital_call":
        return np.exp(-D*T)*Np(d1)/(S*sigma*np.sqrt(T))
    elif option_type=="digital_put":
        return -np.exp(-D*T)*Np(d1)/(S*sigma*np.sqrt(T))
    else:
        raise ValueError("option_type must be either 'call', 'put', 'digital_call' or 'digital_put'")
    
def gamma(S, E, r, D,sigma, T, option_type):
    d1 = d1_func(S, E, r, D,sigma, T)
    if option_type in ["call","put"]:
        return np.exp(-D*T)*Np(d1)/(S*sigma*np.sqrt(T))
    elif option_type in ["digital_call","digital_put"]:
        return -np.exp(-D*T)*d1*Np(d1)/(S**2*sigma**2*T)
    else:
        raise ValueError("option_type must be either 'call', 'put', 'digital_call' or 'digital_put'")
    
