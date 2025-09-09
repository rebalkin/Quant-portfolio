import numpy as np
from scipy.stats import norm

# (norm.cdf)

def d1(S, E, r, D,sigma, T):
    return (np.log(S / E) + (r-D + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
def d2(S, E, r,D, sigma, T):
    return d1(S, E, r, D,sigma, T)-sigma * np.sqrt(T)
def N(x):
    return norm.cdf(x)
def Np(x):
    return norm.pdf(x)

def V(S, E, r, D,sigma, T, option_type):
    if option_type=="call":
        return S*np.exp(-D*T)*N(d1(S,E,r,D,sigma,T))-E*np.exp(-r*T)*N(d2(S,E,r,D,sigma,T))
    elif option_type=="put":
        return E*np.exp(-r*T)*N(-d2(S,E,r,D,sigma,T))-S*np.exp(-D*T)*N(-d1(S,E,r,D,sigma,T))
    elif option_type=="digital_call":
        return np.exp(-r*T)*N(d2(S,E,r,D,sigma,T))
    elif option_type=="digital_put":
        return np.exp(-r*T)*N(-d2(S,E,r,D,sigma,T))
    else:
        raise ValueError("option_type must be either 'call', 'put', 'digital_call' or 'digital_put'")
def V_butterfly(S, E1,E2,E3, r, D,sigma, T):
    return V(S,E1,r,D,sigma,T,"call")-2*V(S,E2,r,D,sigma,T,"call")+V(S,E3,r,D,sigma,T,"call")