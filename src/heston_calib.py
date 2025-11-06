
import pandas as pd
import heston as he
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def load_option_chain(stock,Tmin,Tmax):
    # Fetch option chain data from yfinance
    # stock: ticker symbol string
    # Tmin, Tmax: min and max time to expiry in years

    ticker = yf.Ticker(stock)  
    S0 = ticker.history(period="1d")["Close"].iloc[-1]
    expirations = ticker.options    
    call_data =[]
    put_data =[]
    # Loop through expirations and collect data
    for exp in expirations:
        opt = ticker.option_chain(exp)
        c = opt.calls[['strike','impliedVolatility','bid','ask','lastPrice','volume','openInterest','inTheMoney','lastTradeDate']].copy()
        c['expiration'] = pd.to_datetime(exp) 
        c['mid'] = (c['bid'] + c['ask']) / 2
        c['spread'] = (c['ask'] -c['bid']) 
        c['rel_spread'] = c['spread']/c['mid']
        call_data.append(c)
        p = opt.puts[['strike','impliedVolatility','bid','ask','lastPrice','volume','openInterest','inTheMoney','lastTradeDate']].copy()
        p['expiration'] = pd.to_datetime(exp)
        p['mid'] = (p['bid'] + p['ask']) / 2
        p['spread'] = (p['ask'] -p['bid']) 
        p['rel_spread'] = p['spread']/p['mid']
        put_data.append(p)
        
    df_calls = pd.concat(call_data, ignore_index=True)
    df_puts = pd.concat(put_data, ignore_index=True)

    # Some clean up
    df_calls = df_calls[(df_calls['openInterest'] > 10) | (df_calls['volume'] > 0)]
    df_puts = df_puts[(df_puts['openInterest'] > 10) | (df_puts['volume'] > 0)]
    # Remove options with zero bid or ask
    df_calls = df_calls[(df_calls['bid'] > 0) & (df_calls['ask'] > 0)]
    df_puts = df_puts[(df_puts['bid'] > 0) & (df_puts['ask'] > 0)]

    df_calls['lastTradeDate'] = pd.to_datetime(df_calls['lastTradeDate']).dt.tz_localize(None)
    df_puts['lastTradeDate'] = pd.to_datetime(df_puts['lastTradeDate']).dt.tz_localize(None)

    # Merge
    df = pd.merge(df_calls, df_puts, on=['strike','expiration'], suffixes=('_call','_put'))

    # Add time to expiry
    now = pd.Timestamp.utcnow().tz_localize(None)
    df['T'] = (df['expiration'] - now).dt.total_seconds() / (365.0*24*3600)

    # Add moneyness and forward for risk-free interest rate estimation
    df['moneyness'] = df['strike'] / S0
    df['forward'] = df['mid_call'] - df['mid_put'] + df['strike']
    df['r_forward'] = np.log(df['forward'] / S0) / df['T']
    df['fwd_ratio'] = df['forward'] / S0

    ### Filters for interest rate estimate  -  and not to far from expiry
    moneyness_mask = (df['moneyness'] > 0.95) & (df['moneyness'] < 1.05) # close to money
    close_to_expiry_mask = (df['T'] < 0.5) & (df['T'] > 0.02) # not too close or far from expiry
    forward_ratio_mask = (df['fwd_ratio']>0.999) & (df['fwd_ratio']<1.05) # keep only sensible 0<r<0.1 estimates
    # Calculate average risk-free rate estimate and its std
    r_estimate = df[moneyness_mask & close_to_expiry_mask & forward_ratio_mask]['r_forward'].mean()
    r_error = df[moneyness_mask & close_to_expiry_mask & forward_ratio_mask]['r_forward'].std()
    
    ### Filters for stock option prices

    moneyness_mask_price = np.abs(df['strike']/S0-1)<0.5
    cutoff_days = 30
    last_trade_mask = ((now - df['lastTradeDate_call']).dt.days < cutoff_days)
    T_mask =(df['T'] > Tmin) & (df['T'] < Tmax) 

    price_mask = moneyness_mask_price & last_trade_mask & T_mask

    df = df[price_mask]

    T_unique = df['T'].unique()
    K_unique = df['strike'].unique()
    T_unique.sort()
    K_unique.sort() 

    bid_ask_means =  [np.array(df[df['T']==T]['mid_call']) for T in T_unique]
    spreads =  [np.array(df[df['T']==T]['spread_call']) for T in T_unique]
    k_vals =[np.array(df[df['T']==T]['strike']) for T in T_unique]

    kmin = np.min(np.concatenate(k_vals))
    kmax = np.max(np.concatenate(k_vals))

    return {
        "market_state": {
            "ticker" : stock,
            "S0": S0,
            "r_estimate": r_estimate,
            "r_error": r_error,
        },
        "option_surface": {
            "T_unique": T_unique,
            "k_vals": k_vals,
            "bid_ask_means": bid_ask_means,
            "spreads": spreads,
            "kmin": kmin,
            "kmax": kmax,
        },
    }


def plot_option_price(data,colors):

    T_unique = [d['option_surface']['T_unique'] for d in data]
    k_vals = [d['option_surface']['k_vals'] for d in data]
    bid_ask_means = [d['option_surface']['bid_ask_means'] for d in data]

    minColor = [0.8,0.3]
    maxColor = [0.3,0.8]
    colors_list = [[colors[i](x) for x in np.linspace(minColor[i%2],maxColor[i%2],len(d['option_surface']['T_unique']))] for i,d in enumerate(data)]


    plt.figure(figsize=(20, 8))
    for i in range(len(data)):
        for iT,T in enumerate(T_unique[i]):
            plt.plot(k_vals[i][iT],bid_ask_means[i][iT],label=f'T={T:.2}',color=colors_list[i][iT],marker='o', markersize=2)
    # Assume all data are from same stock, otherwise the titles might be wrong!
    plt.axvline(x=data[0]['market_state']['S0'], color='black', linestyle='--', linewidth=2,alpha=0.5)
    plt.title(f'Call prices for '+data[0]['market_state']['ticker']+ ' stock as a function of strike price for various expiries')
    plt.xlabel('Strike price [$]',fontsize=14)
    plt.ylabel('Option price [$]',fontsize=14)
    plt.legend()
    plt.show()

def residuals(params, k_vals, T_unique, bid_ask_means,spreads, S0, r):
    kappa, theta, sigma, rho, v0 = params
    out = []

    for Ti, T in enumerate(T_unique):
        strikes = k_vals[Ti]              # array of K for this maturity
        market_prices = bid_ask_means[Ti] # same length as strikes
        spreads_T = spreads[Ti]
        u_max=200.0
        n_u=4000
        model_price = he.heston_call_K_vectorized(S0, strikes, v0, r , kappa, theta, sigma, rho, T,u_max, n_u)
        weighted_resid = (model_price - market_prices) / np.maximum(spreads_T, 1e-5)
        out.append(weighted_resid)

    return np.concatenate(out)

def calibrate_heston(data,x0,bounds,verbose=0):

    k_vals =data['option_surface']['k_vals']
    T_unique=data['option_surface']['T_unique']
    bid_ask_means=data['option_surface']['bid_ask_means']
    spreads=data['option_surface']['spreads']
    S0 =data['market_state']['S0']
    r = data['market_state']['r_estimate']
    result = least_squares(residuals,x0,bounds=bounds,args=(k_vals, T_unique, bid_ask_means,spreads, S0, r),verbose=verbose)
    return result

def plot_calibration(data,fits,colors):


    T_unique = [d['option_surface']['T_unique'] for d in data]
    k_vals = [d['option_surface']['k_vals'] for d in data]
    bid_ask_means = [d['option_surface']['bid_ask_means'] for d in data]
    S0 =[d['market_state']['S0'] for d in data]
    r =[d['market_state']['r_estimate'] for d in data]
    kmin =np.min([d['option_surface']['kmin'] for d in data])
    kmax =np.max([d['option_surface']['kmax'] for d in data])

    stock = data[0]['market_state']['ticker'] #Assuming all the data is from the same stock

    minColor = [0.8,0.3]
    maxColor = [0.3,0.8]
    colors_list = [[colors[i](x) for x in np.linspace(minColor[i%2],maxColor[i%2],len(d['option_surface']['T_unique']))] for i,d in enumerate(data)]

    fit_vals = [f.x for f in fits] #kappa, theta, sigma, rho, v0
    u_max=200.0
    n_u=4000
    model_prices = [ ([np.array(he.heston_call_K_vectorized(S0[i], k_vals[i][iT], fit_vals[i][4], r[i] , fit_vals[i][0],fit_vals[i][1], fit_vals[i][2], fit_vals[i][3],T,u_max, n_u)) 
                      for iT,T in enumerate(T_unique[i])]) for i in range(len(data))]

    plt.figure(figsize=(18, 9))
    for i in range(len(data)):
        for iT,T in enumerate(T_unique[i]):
            plt.plot(k_vals[i][iT],np.abs(1- model_prices[i][iT]/bid_ask_means[i][iT]),label =f'T={T:.2}',color=colors_list[i][iT])

    plt.yscale('log')
    plt.xlabel('Strike price [$]',fontsize=14)
    plt.ylabel(r'$|C_{\text{Heston}}/C_{\text{Data}}-1|$',fontsize=14)
    plt.ylim(1e-5, 10) 
    plt.xlim(kmin*0.9, kmax*1.1) 
    plt.title(f'Calibration of Heston model parameters to {stock} stock data')
    # plt.text(0.025, 0.9, 
    #         rf'$S_0=\${S0:.5}$'+"\n"+f'r={100*r:.3}% p.a', 
    #         fontsize=10, color="black", ha="left", va="center",transform=plt.gca().transAxes,bbox=dict(boxstyle="square", facecolor="white", edgecolor="black",alpha=0.8))
    for i in range(len(data)):
        plt.text(0.125+i*0.2, 0.85, 
                f'    Expiry range {np.min(T_unique[i]):.3}<T<{np.max(T_unique[i]):.3}'+'\n'+r'$\sigma_0=$'+f'{np.sqrt(fit_vals[i][4])*100:.4}'+r'% $1/\sqrt{\text{yr}}$'
                +'\n'+r'$\kappa=$'+f'{fit_vals[i][0]:.4}'+'\n'+r'$\sqrt{\theta} =$'+f'{np.sqrt(fit_vals[i][1])*100:.3}'+r'% $1/\sqrt{\text{yr}}$'+ '\n'
                +r'$\sigma=$'+f'{np.sqrt(fit_vals[i][2])*100:.4}'+r'% $1/\sqrt{\text{yr}}$'+'\n'+r'$\rho=$'+f'{fit_vals[i][3]:.3}', 
                fontsize=10, color=colors_list[i][(-1)*(i%2)], ha="left", va="center",transform=plt.gca().transAxes,bbox=dict(boxstyle="square", facecolor="white", edgecolor="black",alpha=0.8))
    # plt.text(0.25, 0.85, 
    #         f'    Long expiry T>{T0}'+'\n'+r'$\sigma_0=$'+f'{np.sqrt(v0_l)*100:.4}'+r'% $1/\sqrt{\text{yr}}$'
    #         +'\n'+r'$\kappa=$'+f'{kappa_l:.4}'+'\n'+r'$\sqrt{\theta} =$'+f'{np.sqrt(theta_l)*100:.3}'+r'% $1/\sqrt{\text{yr}}$'+ '\n'
    #         +r'$\xi=$'+f'{np.sqrt(sigma_l)*100:.4}'+r'% $1/\sqrt{\text{yr}}$'+'\n'+r'$\rho=$'+f'{rho_l:.3}', 
    #         fontsize=10, color=colorsRed[-1], ha="left", va="center",transform=plt.gca().transAxes,bbox=dict(boxstyle="square", facecolor="white", edgecolor="black",alpha=0.8))
    plt.legend(loc='upper right')
    plt.axvline(x=S0[0], color='black', linestyle='--', linewidth=2,alpha=0.5) #Assuming all the data is from the same stock and present stock value
    plt.grid(True,alpha=0.5)
    plt.show()