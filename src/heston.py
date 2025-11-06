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
from numpy.random import noncentral_chisquare
import pandas as pd
from dataclasses import dataclass

from math import log, sqrt, exp, erf

@dataclass
class HestonParams:
    S0: float
    v0: float
    mu: float
    r: float
    KC: float
    KH: float
    TC: float
    TH: float
    eps: float
    u_max: float
    n_u: int
    kappa: float
    theta: float
    lambda_v: float
    sigmavol: float
    rho: float
    n_steps: int
    n_paths: int


def heston_time_series(S0, v0, mu , kappa, theta, sigmavol,rho, T,n_steps,n_paths):

    dt = T/n_steps
    sqrt_dt = np.sqrt(dt);
    Z1 = np.random.normal(size=(n_steps,n_paths))
    Z1perp = np.random.normal(size=(n_steps,n_paths))
    Z2 = rho*Z1+ np.sqrt(1-rho**2)*Z1perp

    Vpath = np.zeros((n_steps + 1,n_paths))
    Spath = np.zeros((n_steps + 1,n_paths))

    Vpath[0] = np.full(n_paths, v0)
    Spath[0] = np.full(n_paths, S0)

    for t in range(1, n_steps + 1):
        v_prev = Vpath[t-1, :]
        v_pos  = np.maximum(v_prev, 0.0)               # clamp before sqrt

        Spath[t] = Spath[t-1, :] * np.exp((mu - 0.5*v_pos)*dt + sqrt_dt*np.sqrt(v_pos)*Z1[t-1, :])
        Vpath[t] = v_pos + kappa*(theta - v_pos)*dt + sqrt_dt*sigmavol*np.sqrt(v_pos)*Z2[t-1, :]

        Vpath[t] = np.maximum(Vpath[t], 0.0)           # clamp after update

    time_grid = np.linspace(0, T, n_steps+1)
    return Vpath.T, Spath.T, time_grid

def heston_call_value_MC(S0, K, v0, mu , kappa, theta, sigmavol,rho, T,n_steps,n_paths):

     Vpath, Spath, time_grid = heston_time_series(S0, v0, mu , kappa, theta, sigmavol,rho, T,n_steps,n_paths)
     price_dist = np.exp(-mu*T)*np.maximum(Spath[:,-1]-K,0)
     call_value = np.mean(price_dist)
     call_value_err = np.std(price_dist)/np.sqrt(n_paths)
     return call_value,call_value_err


def heston_call_value(S0, K, v0, mu , kappa, theta, sigmavol,rho, T):

    i = (1j)

    xi = lambda u: kappa-sigmavol*rho*u*i
    d = lambda u: np.sqrt((xi(u))**2+(sigmavol**2)*(u**2+i*u))
    g2 = lambda u: (xi(u)-d(u))/(xi(u)+d(u))

    phi = lambda u: np.exp(i*u*(np.log(S0)+mu*T)+((kappa*theta)/(sigmavol**2))*((xi(u)-d(u))*T-2*np.log((1-g2(u)*np.exp(-d(u)*T))/(1-g2(u))))+(v0*(xi(u)-d(u))/(sigmavol**2))*((1-np.exp(-d(u)*T))/(1-g2(u)*np.exp(-d(u)*T))))

    int1 = lambda u: np.real(((np.exp(-i*u*np.log(K)))/(i*u*S0*np.exp(mu*T)))*phi(u-i))
    int2 = lambda u: np.real(((np.exp(-i*u*np.log(K)))/(i*u))*phi(u))

    res1, err1 = quad(int1, 0, np.inf)
    res2, err2 = quad(int2, 0, np.inf)

    P1 = 0.5+res1/(np.pi)
    dP1 = err1/(np.pi)

    P2 = 0.5+res2/(np.pi)
    dP2 = err2/(np.pi)

    call_value = S0*P1-np.exp(-mu*T)*K*P2
    call_value_err = np.sqrt((S0*dP1)**2+(np.exp(-mu*T)*K*dP2)**2)

    return call_value, call_value_err

def heston_call_K_vectorized(S0, K, v0, mu , kappa, theta, sigmavol,rho, T,u_max=200.0, n_u=4000):
    # vectorized in K
    i = (1j)
    u = np.linspace(1e-6, u_max, n_u) 
    
    xi = lambda u: kappa-sigmavol*rho*u*i
    d = lambda u: np.sqrt((xi(u))**2+(sigmavol**2)*(u**2+i*u))
    g2 = lambda u: (xi(u)-d(u))/(xi(u)+d(u))

    phi = lambda u: np.exp(i*u*(np.log(S0)+mu*T)+((kappa*theta)/(sigmavol**2))*((xi(u)-d(u))*T-2*np.log((1-g2(u)*np.exp(-d(u)*T))/(1-g2(u))))+(v0*(xi(u)-d(u))/(sigmavol**2))*((1-np.exp(-d(u)*T))/(1-g2(u)*np.exp(-d(u)*T))))

    phi_u         = phi(u)
    phi_u_minus_i = phi(u - i)

    K = np.atleast_1d(K).astype(float)
    logK = np.log(K)

    exp_term = np.exp(-i*np.outer(u, logK))
    denom_u  = (i*u)[:, None]

    integrand1 = np.real(exp_term / (denom_u * S0*np.exp(mu*T)) * phi_u_minus_i[:, None])
    integrand2 = np.real(exp_term / denom_u * phi_u[:, None])

    res1 = np.trapezoid(integrand1, u, axis=0)
    res2 = np.trapezoid(integrand2, u, axis=0)

    P1 = 0.5+res1/(np.pi)

    P2 = 0.5+res2/(np.pi)

    call_value = S0*P1-np.exp(-mu*T)*K*P2

    return call_value

def heston_call_value_vectorized(S0, K, v0, mu , kappa, theta, sigmavol,rho, T,u_max=200.0, n_u=4000):
   
    i = (1j)
    xi = lambda u: kappa-sigmavol*rho*u*i
    d = lambda u: np.sqrt((xi(u))**2+(sigmavol**2)*(u**2+i*u))
    g2 = lambda u: (xi(u)-d(u))/(xi(u)+d(u))

    u_array = np.linspace(1e-6, u_max, n_u) 

    S0 = np.atleast_1d(S0).astype(float)
    v0 = np.atleast_1d(v0).astype(float)
    assert S0.shape == v0.shape, "S0 and v0 must have same length"

    K = np.atleast_1d(K).astype(float)
    if K.size == 1:                       # scalar strike for all pairs
         Kb = float(K[0])                  # keep scalar
         logK = np.log(Kb)                 # scalar
    else:
        assert K.shape == S0.shape, "If K is a vector it must match S0/v0"
        logK = np.log(K)[None, :]         # shape (1, N)

    u  = u_array[:, None]                 # (n_u, 1)
    S0b = S0[None, :]                     # (1, N)
    v0b = v0[None, :]                     # (1, N)


    phi = lambda u,S0,v0: np.exp(i*u*(np.log(S0)+mu*T)+((kappa*theta)/(sigmavol**2))*((xi(u)-d(u))*T-2*np.log((1-g2(u)*np.exp(-d(u)*T))/(1-g2(u))))+(v0*(xi(u)-d(u))/(sigmavol**2))*((1-np.exp(-d(u)*T))/(1-g2(u)*np.exp(-d(u)*T))))

    phi_u         = phi(u,S0b,v0b)
    phi_u_minus_i = phi(u - i,S0b,v0b)


    exp_term = np.exp(-i*u*logK)
    denom_u  = (i*u)

    integrand1 = np.real(exp_term / (denom_u * S0b*np.exp(mu*T)) * phi_u_minus_i)
    integrand2 = np.real(exp_term / denom_u * phi_u)

    res1 = np.trapezoid(integrand1, u_array, axis=0)
    res2 = np.trapezoid(integrand2, u_array, axis=0)

    P1 = 0.5+res1/(np.pi)

    P2 = 0.5+res2/(np.pi)

    call_value = S0 * P1 - np.exp(-mu*T) * (Kb if K.size==1 else K) * P2  # (N,)

    return call_value, P1

def heston_call_greeks(S0, K, v0, mu , kappa, theta, sigmavol,rho, T,eps=1e-3,u_max=200.0, n_u=4000):

    prices,delta = heston_call_value_vectorized(S0, K, v0, mu , kappa, theta, sigmavol,rho, T,u_max, n_u)

    prices_shifted_v0P,_ = heston_call_value_vectorized(S0, K, v0*(1+eps), mu , kappa, theta, sigmavol,rho, T,u_max, n_u)
    prices_shifted_v0M,_ = heston_call_value_vectorized(S0, K, v0*(1-eps), mu , kappa, theta, sigmavol,rho, T,u_max, n_u)
    mask_zero = (v0 < 1e-12)
    vega = np.zeros_like(v0)
    vega[~mask_zero] = (
        (prices_shifted_v0P[~mask_zero] - prices_shifted_v0M[~mask_zero]) /
        (2 * eps * v0[~mask_zero])
    )

    return prices, delta, vega




def heston_vol_surface(S0, K_range, v0, mu , kappa, theta, sigmavol,rho, T_range,u_max=200.0, n_u=4000,tol=1e-8, max_iter=50, sigma0=0.2, sigma_min=1e-8, sigma_max=5.0):

    T_vals = np.atleast_1d(T_range)
    surface = np.zeros((T_vals.shape[0],K_range.shape[0]))
    for i,T in enumerate(T_vals):
        heston_call_price = heston_call_value_vectorized(S0, K_range, v0, mu , kappa, theta, sigmavol,rho, T,u_max, n_u)
        surface[i] = bs.implied_vol_newton(S0, K_range, mu, 0, T, 'call', heston_call_price,tol, max_iter, sigma0, sigma_min, sigma_max)

    return surface

def heston_hedging(params: HestonParams):


    S0 = params.S0
    v0 = params.v0
    mu= params.mu
    r= params.r
    KC= params.KC
    KH= params.KH
    TC= params.TC
    TH= params.TH
    eps= params.eps
    u_max= params.u_max
    n_u= params.n_u
    kappa= params.kappa
    theta= params.theta
    lambda_v= params.lambda_v
    sigmavol= params.sigmavol
    rho= params.rho
    n_steps= params.n_steps
    n_paths= params.n_paths

    assert TH>=TC, "H must not expire before C"
    kappaP = kappa - lambda_v       #physical measure kappa
    thetaP = theta * (kappa / (kappa - lambda_v)) #physical measure theta

    Vpaths, Spaths, time_grid = heston_time_series(S0, v0, mu , kappaP, thetaP, sigmavol,rho, TC,n_steps,n_paths)
    time_to_exp_C = TC-time_grid
    time_to_exp_H = TH-time_grid

    records = []
    for i in range(len(time_to_exp_C)):
        St = Spaths[:, i]
        vt = Vpaths[:, i]
        prices_C, delta_C, vega_C = heston_call_greeks(St, KC, vt, r, kappa, theta, sigmavol, rho, time_to_exp_C[i],eps,u_max, n_u)
        prices_H, delta_H, vega_H = heston_call_greeks(St, KH, vt, r, kappa, theta, sigmavol, rho, time_to_exp_H[i],eps,u_max, n_u)
        for j in range(len(St)):
            records.append({
                "path": j,
                "time_index": i,
                "time": time_grid[i],
                "S": St[j],
                "v": vt[j],
                "discount": np.exp(-r*time_grid[i]),
                "price_C": prices_C[j],
                "delta_C": delta_C[j],
                "vega_C": vega_C[j],
                "price_H": prices_H[j],
                "delta_H": delta_H[j],
                "vega_H": vega_H[j],
                "time_expiry_H": time_to_exp_H[i],
                "time_expiry_C": time_to_exp_C[i],
            })

    df = pd.DataFrame.from_records(records)

    df['phi'] = df['vega_C']/df['vega_H']
    df['delta'] = df['delta_C']-df['phi']*df['delta_H']

    initial_pos = df[df['time']==0]
    B0 = (-initial_pos['price_C']+initial_pos['delta']*S0+initial_pos['phi']*initial_pos['price_H']).values

    df["dphi"]   = df.groupby("path")["phi"].diff()
    df["ddelta"] = df.groupby("path")["delta"].diff()
    df["cashflow"] = (df['ddelta']*df['S']+df['dphi']*df['price_H'])*df['discount']
    df = df.sort_values(["path","time"])
    cf = (df.ddelta*df.S + df.dphi*df.price_H) * df.discount

    g = df.groupby("path")
    mask = g.cumcount().between(1, g.transform("size") - 2)
    cashflow_by_path = (cf[mask].groupby(df["path"]).sum()).values

    final_pos = df[df['time']==TC]
    final_delta = df[df['time_index']==len(time_to_exp_C)-2]['delta']
    final_phi = df[df['time_index']==len(time_to_exp_C)-2]['phi']

    if (TH>TC):
        H_payoff = final_pos['price_H']
    elif (TH==TC):
        H_payoff = np.maximum(final_pos['S']-KH,0)

    BN = (np.exp(-r*TC)*(np.maximum(final_pos['S']-KC,0)-final_delta.values*(final_pos['S'].values)-final_phi.values*(H_payoff.values))).values
    PnL = B0+cashflow_by_path+BN
    PnL_no_hedge = -initial_pos['price_C'].values+np.exp(-r*TC)*np.maximum(final_pos['S']-KC,0).values

    B0_delta = (-initial_pos['price_C']+initial_pos['delta']*S0).values
    cf_delta = (df.ddelta*df.S ) * df.discount
    cashflow_by_path_delta = (cf_delta[mask].groupby(df["path"]).sum()).values
    BN_delta = (np.exp(-r*TC)*(np.maximum(final_pos['S']-KC,0)-final_delta.values*(final_pos['S'].values))).values

    PnL_delta = cashflow_by_path_delta+B0_delta+BN_delta

    return PnL_no_hedge,PnL_delta,PnL,df


