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

import numpy as np

def eval_option_tree_fast(S0, r, q, sigma, T, N_steps, payoff, exercise_type, K=None):
    """
    Vectorized binomial tree over S0, T, sigma, and optionally K.

    Inputs:
      S0:    scalar or (m,)
      r, q:  scalars (can be arrays broadcastable to T/sigma if desired)
      sigma: scalar or (p,*)  (any shape broadcastable with T)
      T:     scalar or (p,*)  (must be broadcastable with sigma)
      N_steps: int
      payoff: function(ST, K) -> same-shape array  (if K=None, accepts payoff(ST))
      exercise_type: "european" or "american"
      K:     scalar or (k,)   (optional)

    Returns:
      Shapes:
        no K:  (m, P)  where P = np.prod(broadcast_shape_of(T, sigma))
        with K: (m, P, k)
      Scalars are squeezed.
    """
    is_american = (exercise_type.lower() == "american")

    # Shapes
    S0 = np.atleast_1d(S0)                 # (m,)
    m = S0.size

    # Broadcast T and sigma (and optionally r, q) to a common 1D param axis P
    T  = np.asarray(T)
    sigma = np.asarray(sigma)

    if T.ndim == 1 and sigma.ndim == 1 and T.shape != sigma.shape:
    # interpret as full grid of all T×sigma combinations
        T     = T[:, None]      # (t,1)
        sigma = sigma[None, :]  # (1,s)


    # Compute the common broadcast shape, then flatten to (P,)
    bshape = np.broadcast_shapes(T.shape, sigma.shape)
    T  = np.broadcast_to(T,  bshape).ravel()            # (P,)
    sigma = np.broadcast_to(sigma, bshape).ravel()      # (P,)
    P = T.size

    # Allow r, q to broadcast too if arrays were passed
    r = np.broadcast_to(np.asarray(r), bshape).ravel()
    q = np.broadcast_to(np.asarray(q), bshape).ravel()

    # Per-param step quantities
    dt   = T / N_steps                                   # (P,)
    u    = np.exp(sigma * np.sqrt(dt))                   # (P,)
    v    = np.exp(-sigma * np.sqrt(dt))                  # (P,)
    p    = (np.exp((r - q) * dt) - v) / (u - v)          # (P,)
    disc = np.exp(-r * dt)                               # (P,)

    # Powers for all nodes (node axis = N+1, param axis = P)
    i = np.arange(N_steps + 1)
    u_pow = u[None, :] ** (N_steps - i)[:, None]         # (N+1, P)
    v_pow = v[None, :] ** (i)[:, None]                   # (N+1, P)

    # Terminal stock prices ST: (m, N+1, P)
    ST = S0[:, None, None] * u_pow[None, :, :] * v_pow[None, :, :]

    # Terminal payoff (optionally over K -> last axis k)
    if K is None:
        cur = payoff(ST)                                 # (m, N+1, P)
    else:
        K = np.atleast_1d(K)                             # (k,)
        cur = payoff(ST[..., None], K[None, None, None, :])  # (m, N+1, P, k)

    # Rollback along node axis
    if K is None:
        # (m, n+1, P)
        while cur.shape[1] > 1:
            cur = disc[None, None, :] * (
                p[None, None, :] * cur[:, :-1, :] + (1 - p)[None, None, :] * cur[:, 1:, :]
            )
            if is_american:
                n_now = cur.shape[1] - 1
                j = np.arange(n_now + 1)
                u_now = u[None, :] ** (n_now - j)[:, None]      # (n_now+1, P)
                v_now = v[None, :] ** (j)[:, None]              # (n_now+1, P)
                ST_now = S0[:, None, None] * u_now[None, :, :] * v_now[None, :, :]
                cur = np.maximum(cur, payoff(ST_now))
        out = cur[:, 0, :]                                      # (m, P)
    else:
        # (m, n+1, P, k)
        while cur.shape[1] > 1:
            cur = disc[None, None, :, None] * (
                p[None, None, :, None] * cur[:, :-1, :, :] + (1 - p)[None, None, :, None] * cur[:, 1:, :, :]
            )
            if is_american:
                n_now = cur.shape[1] - 1
                j = np.arange(n_now + 1)
                u_now = u[None, :] ** (n_now - j)[:, None]      # (n_now+1, P)
                v_now = v[None, :] ** (j)[:, None]              # (n_now+1, P)
                ST_now = S0[:, None, None] * u_now[None, :, :] * v_now[None, :, :]
                cur_pay = payoff(ST_now[..., None], K[None, None, None, :])
                cur = np.maximum(cur, cur_pay)
        out = cur[:, 0, :, :]                                   # (m, P, k)

    return np.squeeze(out)

 




