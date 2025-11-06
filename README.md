# Quant finance portfolio

This repository showcases my work in **quantitative finance**, applying numerical methods and simulations.  
It demonstrates my ability to **model stochastic processes**, **implement algorithms from scratch**, and **analyze convergence and error** - skills directly relevant to quantitative research and trading.

---

## 🧩 Skills Demonstrated

- **Stochastic Modeling:** Geometric Brownian Motion (GBM) for asset price simulation.  
- **Numerical Methods:** Binomial tree pricing, Monte Carlo simulation, error analysis.  
- **Analytical Finance:** Black–Scholes closed-form benchmarks, Greeks, payoff structures.  
- **Computational Skills:** Python, NumPy, SciPy, Matplotlib, Jupyter.  
- **Research Mindset:** Comparing methods, analyzing convergence rates, validating against theory.  

---

## 📂 Project Highlights

### 1. [Option Pricing Methods](/notebooks/option_pricing.ipynb) (Binomial Trees, Monte Carlo, Black–Scholes)

- Implemented and compared three core approaches to option pricing:
  - **Binomial trees** (American vs. European calls/puts, convergence to Black–Scholes).
  - **Black–Scholes benchmarks** (closed-form solutions, butterfly spread payoff).
  - **Monte Carlo simulation** (GBM under risk-neutral measure, confidence intervals, error scaling).
- Analyzed sensitivities to strike, maturity, and volatility, highlighting convergence and early exercise effects.
  
👉 Skills: stochastic simulation, numerical methods, analytical finance, error analysis, visualization.

### 2. [Volatility Estimation](notebooks/volatility_estimation.ipynb) (Historical, Rolling, EWMA)

- Implemented and compared multiple approaches to volatility estimation:
  - **Historical volatility** (using log returns, annualization).
  - **Rolling window estimates** (sliding windows to track changing variance).
  - **Exponentially Weighted Moving Average (EWMA)** (linking half-life to decay factor λ).
- Visualized and compared methods across different time periods, highlighting responsiveness to market shocks and the trade-off between smoothness and reactivity.

👉 Skills: time-series modeling, statistical finance, stochastic processes, data visualization.

### 3. [Volatility Arbitrage Simulation](notebooks/vol_arb.ipynb) (Hedging, Error Analysis, Monte Carlo)  
   - Implemented and analyzed a volatility arbitrage strategy by simulating option hedging under **different assumed volatilities**.  
   - Developed functions for option pricing and hedging P&L using:  
     - **Black–Scholes model** (closed-form benchmark for European options).  
     - **Monte Carlo simulations** (stock price paths under GBM, risk-neutral valuation).  
     - **Discrete re-hedging** (examined convergence to continuous limit, error scaling).  
   - Quantified the effect of **mismatch between hedging volatility and true volatility**, highlighting sources of arbitrage profit/loss.  
   - Analyzed sensitivity to hedge interval, including statistical error estimates.  

   👉 **Skills:** stochastic calculus, Monte Carlo simulation, risk-neutral pricing, error analysis, numerical methods, financial engineering.

### 4. [Heston Calibration & Hedging Simulation](notebooks/heston.ipynb) (Stochastic Volatility, Calibration, Hedging Error)  
   - Calibrated the **Heston model** to full implied-volatility surfaces (AMZN, GOOG, AMD).  
   - Simulated **Heston price/variance paths under the real-world measure \(P\)** while computing Greeks under the **risk-neutral measure \(Q\)** to study the impact of the volatility risk premium \( \lambda_v \).  
   - Implemented **delta and delta+vega hedging** using Heston Greeks and analyzed discrete-time hedging error.  

   👉 **Skills:** Heston model, stochastic volatility, calibration & optimization, Monte Carlo simulation, hedging, Greeks, numerical methods, financial engineering.



---

## 📊 Example Figures

**American vs. European Put Prices**

<img src="plots/put_vs_S0.png" alt="Put vs Strike" width="500"/>

**ITM/OTM Put Options as a function of expiry**

<img src="plots/put_vs_Sigma.png" alt="Butterfly Spread" width="500"/>

**Monte Carlo Convergence with Error Bars**

<img src="plots/call_vs_S0_conv.png" alt="MC Convergence" width="500"/>

---
