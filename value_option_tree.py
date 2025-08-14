import numpy as np

def call_payoff(S,E):
    # Calculate the payoff of a European call option
    # S: Current stock price
    # E: Strike price
    return(np.maximum(S-E,0))

def eval_option_1_step(V_plus,V_minus,r,p,dt):
    # Evaluate the value of a European call option at one time step
    # V_plus: Value of the option if the stock price goes up
    # V_minus: Value of the option if the stock price goes down
    # r: Risk-free interest rate
    # p: Risk-neutral probability of the stock price going up
    # dt: Time step size
    # Returns the value of the option at the current time step
    
    return (p*V_plus+(1-p)*V_minus)/(1+r*dt) 


def eval_option_1_layer(vals,r,p,dt):
    # Evaluate the value of a European call option over one layer of option prices
    # vals: Array of option prices at the current layer
    # r: Risk-free interest rate
    # p: Risk-neutral probability of the stock price going up
    # dt: Time step size
    # Returns an array of option prices at the next layer
    N = vals.shape[0]
    result = np.zeros((N-1,1))
    for i in range(N-1):
        result[i] = eval_option_1_step(vals[i], vals[i+1], r, p, dt);
    return result

def eval_option_tree(S,E,r,sigma,expiry,N_steps):
    # Evaluate the value of a European call option using a binomial tree
    # S: Current stock price
    # E: Strike price
    # r: Risk-free interest rate
    # sigma: Volatility of the stock price
    # expiry: Time to expiration in years
    # N_steps: Number of time steps in the binomial tree
    # Returns the value of the option at the current time step
    dt = expiry/N_steps  # Time step size
    u = np.exp(sigma*np.sqrt(dt))  # Up factor
    v = np.exp(-sigma*np.sqrt(dt))  # Down factor
    p = (np.exp(r*dt) - v) / (u - v) # Risk-neutral probability of the stock price going up
    # u = 1.0604
    # v = 0.9431
    # p = 0.5567
    cur_vals = np.zeros((N_steps+1,1))
    for i in range(N_steps+1):

        cur_vals[i] = call_payoff(S*u**(N_steps-i)*v**i,E)
    cur_layer = N_steps+1;
    while cur_layer>1:
        # print(cur_vals)
        cur_vals = eval_option_1_layer(cur_vals,r,p,dt)
        cur_layer = cur_layer-1
    return cur_vals[0,0]




