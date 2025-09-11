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

### 1. Binomial Tree Pricing
- American vs. European call and put options.  
- Pricing as a function of strike and volatility.  
- Demonstrated convergence of binomial estimates to the Black–Scholes formula.  

👉 *Skills: algorithm implementation, numerical convergence, visualization.*

---

### 2. Black–Scholes Benchmarks
- Implemented closed-form call/put option formulas.  
- Compared analytic solutions with discrete binomial approximations.  
- Modeled complex payoffs like the **butterfly spread**.  

👉 *Skills: mathematical finance, model validation, payoff engineering.*

---

### 3. Monte Carlo Simulation
- Simulated stock paths using **GBM under the risk-neutral measure**.  
- Estimated European option prices by averaging discounted payoffs.  
- Added **statistical rigor** with confidence intervals and error scaling (\(1/\sqrt{N}\)).  

👉 *Skills: stochastic simulation, variance reduction, statistical analysis.*

---

## 📊 Example Figures

**American vs. European Put Prices**

<img src="plots/put_vs_S0.png" alt="Put vs Strike" width="500"/>

**ITM/OTM Put Options as a function of expiry**

<img src="plots/put_vs_Sigma.png" alt="Butterfly Spread" width="500"/>

**Monte Carlo Convergence with Error Bars**

<img src="plots/call_vs_S0_conv.png" alt="MC Convergence" width="500"/>

---
