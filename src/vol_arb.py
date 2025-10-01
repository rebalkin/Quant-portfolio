import sys
sys.path.append('../src')
import monte_carlo as mc
import black_scholes as bs
import numpy as np
import payoff as po
from scipy.stats import norm
from scipy import integrate
from scipy.optimize import brentq
from scipy.integrate import quad

def vol_arb(S0,r,K,D,sigmaH,sigmaReal,sigmaImp,T,dt,MC,MC_given):
    # S0: spot price
    # r: risk-free rate
    # K: strike price
    # D: continuous dividend yield
    # sigmaH: volatility used for hedging 
    # sigmaReal: real world volatility used to simulate stock price paths
    # sigmaImp: implied volatility used to price the option
    # T: time to maturity
    # dt: time step for rehedging
    # MC: if MC_given is False, MC is a tuple (mu,m) where mu is the drift used to simulate stock price paths and m is the number of simulations
    #     if MC_given is True, MC is a tuple (timegrid,stocks) where timegrid is a 1D array of time points and stocks is a 2D array of simulated stock prices
    # MC_given: boolean indicating whether MC is given as (mu,m) or (timegrid,stocks)
    # Returns: PL_final, percentiles of PL_final at -2SD, -1SD, median, +1SD, +2SD
    # PL_final: 1D array of final profit and loss from the hedging strategy
    # percentiles: 1D array of percentiles of PL_final at -2SD, -1SD, median, +1SD, +2SD
    percentiles = [norm.cdf(x)*100 for x in range(-2,3)]

               # times t0,...,t_n with t_n = T
   
    # n = int(round(T / dt))
    if MC_given==False: 
        #If MC is not given, MC specifies the drift and number of simulations
        mu,m = MC
        timegrid,stocks = mc.GBM_time_series_fast(S0, mu, sigmaReal, T,dt,m)
        stocks = stocks.T
        time2expiry = T-timegrid
        time2expiry = np.tile(time2expiry, (m, 1))
    else:
        timegrid,stocks = MC 
        stocks = stocks.T
        m = stocks.shape[0]
        time2expiry = T-timegrid
        time2expiry = np.tile(time2expiry, (m, 1))
        # Or MC can already be an output of mc.GBM_time_series_fast(S0, mu, sigma, T,dt,m)
        # This can be used to avoide regenerating tracks if (S0, mu, sigma, T,dt,m) are unchanged

    n = stocks.shape[1] - 1  
    t = timegrid  

    delta0 = bs.delta(S0,K,r,D,sigmaH,T,'call')
    B0 = -bs.bs_formula_price(S0,K,r,D,sigmaImp,T,'call')+delta0*S0
    delta = bs.delta(stocks[:,:-1],K,r,D,sigmaH,time2expiry[:,:-1],'call')
    deltaDiff = np.diff(delta,axis=1)
    cashSteps = deltaDiff*stocks[:,1:-1]
    # cashExpWeights =  np.exp(r * dt * (n - np.arange(1, n)))
    cashExpWeights = np.exp(r * (T - t[1:-1])) 
    cashAccFromStock = np.sum(cashSteps*cashExpWeights,axis=1)
    cash_final = cashAccFromStock + B0*np.exp(r*dt*n)
    payoff_final = po.payoff_call(stocks[:,-1],K)
    stock_final = -delta[:,-1]*stocks[:,-1]
    PL_final = (cash_final+payoff_final+stock_final)*np.exp(-r*T) #discount to today
    mean_std = np.std(PL_final)/np.sqrt(m)
    return PL_final, np.array([np.percentile(PL_final,p) for p in percentiles]), mean_std


def vol_arb_average_PL(S0,r,K,D,sigmaH,sigma,sigmaImp,mu,T):

    alpha = lambda t: np.log(S0/K)+(mu-0.5*sigma**2)*t+(r-D+0.5*sigmaH**2)*(T-t)


    integrand = lambda t: ( (S0*np.exp((mu-D-0.5*sigma**2)*t))/(np.sqrt(2*np.pi*((T-t)*sigmaH**2+t*sigma**2)))
                           *np.exp(-0.5*(alpha(t)**2+2*alpha(t)*sigma**2*t-sigma**2*sigmaH**2*t*(T-t))/((T-t)*sigmaH**2+t*sigma**2)) )

    val, err = quad(integrand, 0.0, T, limit=1000)

    PL_average = (bs.bs_formula_price(S0, K, r, D,sigmaH, T, 'call')
                  -bs.bs_formula_price(S0, K, r, D,sigmaImp, T, 'call')
                  +0.5*(sigma**2-sigmaH**2)*val)
    
    PL_error = 0.5*(sigma**2-sigmaH**2)*err
    
    return PL_average, PL_error




    

