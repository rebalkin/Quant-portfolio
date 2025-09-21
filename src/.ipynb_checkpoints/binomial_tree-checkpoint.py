import numpy as np

def call_payoff(S,E):
    # Calculate the payoff of a call option
    # S: Current stock price
    # E: Strike price
    return(np.maximum(S-E,0))

def put_payoff(S,E):
    # Calculate the payoff of a put option
    # S: Current stock price
    # E: Strike price
    return(np.maximum(E-S,0))

def eval_option_1_step(V_plus,V_minus,r,p,dt):
    # Evaluate the value of a European call option at one time step
    # V_plus: Value of the option if the stock price goes up
    # V_minus: Value of the option if the stock price goes down
    # r: Risk-free interest rate
    # p: Risk-neutral probability of the stock price going up
    # dt: Time step size
    # Returns the value of the option at the current time step
    
    return np.exp(-r * dt)*(p*V_plus+(1-p)*V_minus)

## cur_vals,r,p,dt,option_type,exercise_type,cur_layer,u,v,S0,E
def eval_option_1_layer(vals,r,p,dt,option_type,exercise_type,cur_layer,u,v,S0,E):
    # Evaluate the value of a European call option over one layer of option prices
    # vals: Array of option prices at the current layer
    # r: Risk-free interest rate
    # p: Risk-neutral probability of the stock price going up
    # dt: Time step size
    # exercise_type is either "european" or "american"
    # cur_layer: the layer under consideration
    # u: Up factor
    # v: Down factor
    # S0: Current stock price
    # Returns an array of option prices at the next layer
    N = vals.shape[0]
    result = np.zeros((N-1,1))
    for i in range(N-1):      
        risk_free_val = eval_option_1_step(vals[i], vals[i+1], r, p, dt)
        if exercise_type=="european":
            result[i] = risk_free_val
        else:
            cur_stock_price = S0*u**(cur_layer-2-i)*v**i
            if option_type=="call":
                exercise_val = call_payoff(cur_stock_price,E)
            else:
                exercise_val = put_payoff(cur_stock_price,E)
            result[i] = np.maximum(risk_free_val,exercise_val) 
    return result

def eval_option_tree(S0,E,r,sigma,expiry,N_steps,option_type,exercise_type):
    # Evaluate the value of a European call or put option using a binomial tree
    # S: Current stock price
    # E: Strike price
    # r: Risk-free interest rate
    # sigma: Volatility of the stock price
    # expiry: Time to expiration in years
    # N_steps: Number of time steps in the binomial tree
    # option_type is either "call" or "put"
    # exercise_type is either "european" or "american"
    # Returns the value of the option at the current time step
    dt = expiry/N_steps  # Time step size
    u = np.exp(sigma*np.sqrt(dt))  # Up factor
    v = np.exp(-sigma*np.sqrt(dt))  # Down factor
    p = (np.exp(r*dt) - v) / (u - v) # Risk-neutral probability of the stock price going up
    cur_vals = np.zeros((N_steps+1,1))
    for i in range(N_steps+1):
        if option_type=="put":
            cur_vals[i] = put_payoff(S0*u**(N_steps-i)*v**i,E)
        else:
            cur_vals[i] = call_payoff(S0*u**(N_steps-i)*v**i,E)
    cur_layer = N_steps+1;
    while cur_layer>1:
        # print("Layer: ",cur_layer, "values are: ",cur_vals.T)
        cur_vals = eval_option_1_layer(cur_vals,r,p,dt,option_type,exercise_type,cur_layer,u,v,S0,E)
        cur_layer = cur_layer-1
    # print("Layer: ",cur_layer, "values are: ",cur_vals.T)
    return cur_vals[0,0]

def eval_option_tree_fast(S0, r, q, sigma, T, N_steps, payoff, exercise_type, K=None):
    """
    Vectorized binomial tree:
      S0: scalar or (m,)
      K:  scalar or (k,) — optional (if None, payoff must accept payoff(ST) only)
      payoff: function taking payoff(ST, K) and returning same-shape array
              e.g. lambda ST, K: np.maximum(ST - K, 0.0)
    Returns:
      If S0 and K are scalars -> float
      If S0 is (m,) and K is scalar/None -> (m,)
      If S0 is scalar and K is (k,) -> (k,)
      If S0 is (m,) and K is (k,) -> (m, k)
    """
    dt = T / N_steps
    u = np.exp(sigma * np.sqrt(dt))
    v = np.exp(-sigma * np.sqrt(dt))
    p = (np.exp((r - q) * dt) - v) / (u - v)
    disc = np.exp(-r * dt)
    is_american = (exercise_type.lower() == "american")

    scalar_S0 = np.ndim(S0) == 0
    S0 = np.atleast_1d(S0)                 # (m,)
    m = S0.size

    # Terminal stock prices S_T for each S0 across N_steps+1 nodes
    i = np.arange(N_steps + 1)
    ST = S0[:, None] * (u ** (N_steps - i)) * (v ** i)   # (m, N+1)

    if K is None:
        # Payoff signature: payoff(ST)
        cur_vals = payoff(ST)                              # (m, N+1)
    else:
        K = np.atleast_1d(K)                               # (k,)
        # Broadcast payoff over strikes
        cur_vals = payoff(ST[..., None], K[None, None, :]) # (m, N+1, k)

    # Roll back
    if K is None:
        # shapes: (m, n+1)
        while cur_vals.shape[1] > 1:
            cur_vals = disc * (p * cur_vals[:, :-1] + (1 - p) * cur_vals[:, 1:])
            if is_american:
                cur_N = cur_vals.shape[1] - 1
                j = np.arange(cur_N + 1)
                ST_now = S0[:, None] * (u ** (cur_N - j)) * (v ** j)  # (m, cur_N+1)
                cur_vals = np.maximum(cur_vals, payoff(ST_now))        # (m, cur_N+1)
        out = cur_vals[:, 0]                                           # (m,)
    else:
        # shapes: (m, n+1, k)
        while cur_vals.shape[1] > 1:
            cur_vals = disc * (p * cur_vals[:, :-1, :] + (1 - p) * cur_vals[:, 1:, :])
            if is_american:
                cur_N = cur_vals.shape[1] - 1
                j = np.arange(cur_N + 1)
                ST_now = S0[:, None] * (u ** (cur_N - j)) * (v ** j)   # (m, cur_N+1)
                cur_payoff = payoff(ST_now[..., None], K[None, None, :])  # (m, cur_N+1, k)
                cur_vals = np.maximum(cur_vals, cur_payoff)
        out = cur_vals[:, 0, :]                                        # (m, k)

    # Return scalars/vectors with nice shapes
    if scalar_S0 and (K is None or np.ndim(K) == 0):
        return float(np.squeeze(out))
    return np.squeeze(out)
 




