<div align="center">

# ₿ Crypto Risk Profiler Terminal

### *Know Your Risk. Protect Your Bag.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Terminal%20%2F%20CLI-black?style=for-the-badge&logo=windowsterminal&logoColor=white)](https://github.com/Kazuka23)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)]()
[![CRYPTO](https://img.shields.io/badge/KZ%PROJECT-2026-blueviolet?style=for-the-badge)]()

<br/>

> A quantitative CLI tool that helps crypto investors understand, measure, and manage their portfolio risk — right from the terminal.

<br/>

![banner](https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,50:16213e,100=0f3460&height=120&section=header&text=Crypto%20Risk%20Profiler&fontColor=00d4ff&fontSize=36&fontAlignY=65&animation=fadeIn)

</div>

---

## 📌 The Problem

Every day, thousands of young investors dive into crypto driven by **FOMO** — Fear Of Missing Out. They buy high, panic sell low, and never truly understand *why* their portfolio bled out.

The hard truth? Most retail crypto investors have **zero quantitative framework** for evaluating risk. They don't know their volatility. They've never calculated a Sharpe Ratio. And they have no plan for when the market crashes 30% overnight.

**Crypto Risk Profiler Terminal** was built to change that.

---

## 💡 What It Does

This is a fully interactive, terminal-based Python application that analyses your cryptocurrency portfolio using the same mathematical tools used by professional quant traders — packaged in a clean, beginner-friendly CLI experience.

| Feature | Description |
|---|---|
| 📊 **Portfolio Valuation** | Fetches live prices via Yahoo Finance and calculates your total portfolio value in USD |
| 📉 **Annualized Volatility** | Measures how wildly your assets swing in price over a year |
| ⚡ **Sharpe Ratio** | Quantifies how much return you earn *per unit of risk* |
| 💥 **Historical Crash Test** | Simulates a real-world 30% market drop (validated by May 2021, LUNA collapse, FTX) |
| 🛡️ **Risk Profiling** | Scores your portfolio and classifies it: **Conservative**, **Moderate**, or **Aggressive Degen** |
| 🔄 **Rebalancing Recommendations** | If you're overexposed, it generates a precise stablecoin rebalancing plan |

---

## 🧠 The Math Behind It

```
Annualized Volatility  =  σ_daily  ×  √365
Annualized Return      =  (1 + μ_daily)^365  −  1
Sharpe Ratio           =  (R_portfolio − R_free) / σ_portfolio
Crash Loss             =  Portfolio_Value × 0.30
```

> Risk-Free Rate is set to **0%** — because in a pure crypto portfolio, there is no risk-free asset.

---

## 🛠️ Built With

<div align="center">

[![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Yahoo Finance](https://img.shields.io/badge/yfinance-6001D2?style=for-the-badge&logo=yahoo&logoColor=white)](https://pypi.org/project/yfinance/)
[![Rich](https://img.shields.io/badge/Rich%20UI-Terminal-black?style=for-the-badge&logo=windowsterminal&logoColor=white)](https://github.com/Textualize/rich)

</div>

---

## 🚀 How to Use

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Kazuka23/crypto-risk-profiler.git
cd crypto-risk-profiler
```

### 2️⃣ Install Dependencies

```bash
pip install yfinance rich pandas numpy
```

> ⚠️ Requires **Python 3.10 or higher**. Check your version with `python --version`.

### 3️⃣ Run the App

```bash
python app.py
```

### 4️⃣ Follow the Terminal Prompts

```
🪙 Enter coin symbol (e.g. BTC, ETH, SOL) or 'done': BTC
📦 How many BTC do you hold? 0.5

🪙 Enter coin symbol (e.g. BTC, ETH, SOL) or 'done': ETH
📦 How many ETH do you hold? 3

🪙 Enter coin symbol (e.g. BTC, ETH, SOL) or 'done': done
```

The app will fetch live data, crunch the numbers, and display your full risk report — all inside the terminal. 🎯

---

## 📁 Project Structure

```
crypto-risk-profiler/
│
├── app.py          # Main application — all logic and UI in one file
├── README.md       # You are here
└── LICENSE         # MIT License
```

---

## 👨‍💻 Creator

<div align="center">

[![GitHub](https://img.shields.io/badge/Made%20by-Kazuka23-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Kazuka23)

Built for the **relieve boredom**

</div>

---

## 📜 License

```
MIT License — Copyright (c) 2026 Kazuka23

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, subject to the conditions of the MIT License.
```

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

---

## 🌐 Let's Connect

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sya-ban-ibrahim-300ab53a5?utm_source=share_via&utm_content=profile&utm_medium=member_android)
[![Instagram](https://img.shields.io/badge/Instagram-Follow-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://www.instagram.com/kazuwascooked)
[![Line](https://img.shields.io/badge/LINE-Chat-00C300?style=for-the-badge&logo=line&logoColor=white)](https://line.me/ti/p/ELnDqL-72e)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:syabanmuhammad12@gmail.com)

</div>

---

## ⭐ Support This Project

If this project helped you understand your crypto risk better, consider giving it a star — it means the world and helps others discover the tool!

<div align="center">

[![Star this repo](https://img.shields.io/github/stars/Kazuka23/crypto-risk-profiler?style=social)](https://github.com/Kazuka23/crypto-risk-profiler)

**👆 Click the Star button at the top of this page!**

</div>

### ☕ Buy Me a Coffee

If you'd like to support development and keep this project alive:

<div align="center">

[![Saweria](https://img.shields.io/badge/Donate-Saweria-orange?style=for-the-badge&logo=buymeacoffee&logoColor=white)](https://saweria.co/siryagami)

> 💛 Every contribution, no matter how small, is deeply appreciated.

</div>

---

<div align="center">

![footer](https://capsule-render.vercel.app/api?type=waving&color=0:0f3460,50:16213e,100:1a1a2e&height=100&section=footer)

Made with ❤️

</div>
