# Quant Training: Option Pricing Methods

A collection of numerical methods for option pricing, developed while studying quantitative finance.  
The repo explores **binomial trees**, **Monte Carlo simulation**, and **Black–Scholes benchmarks**, with a focus on comparing methods, analyzing convergence, and visualizing option behavior.

---

## 📂 Contents

### 1. Binomial Trees
- Pricing American and European calls and puts.
- Option prices vs. strike and volatility.
- Convergence of binomial tree estimates to the Black–Scholes formula.

### 2. Black–Scholes Benchmarks
- Closed-form solutions for European calls and puts.
- Payoff comparisons at expiry.
- Example of a **butterfly spread** payoff and pricing.

### 3. Monte Carlo Simulation
- Simulating stock paths via **Geometric Brownian Motion**.
- Monte Carlo estimation of European option prices.
- Convergence of simulation estimates to Black–Scholes, with confidence intervals.
- Error scaling ∝ \(1/\sqrt{N}\).

---

## 📊 Example Figures

### American vs European Put Prices
<img src="plots/binomial/put_vs_strike.png" alt="Put vs Strike" width="500"/>

### Butterfly Spread under Black–Scholes
<img src="plots/black_scholes/butterfly.png" alt="Butterfly Spread" width="500"/>

### Monte Carlo Convergence with Error Bars
<img src="plots/monte_carlo/convergence.png" alt="MC Convergence" width="500"/>
