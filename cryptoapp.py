# ──────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

warnings.filterwarnings("ignore")

# Global Rich console — all terminal output routes through this object
console = Console()


# ──────────────────────────────────────────────────────────────────
# SECTION A: CONSTANTS & CONFIGURATION
# ──────────────────────────────────────────────────────────────────

# Crypto markets trade every day of the year — unlike equities (252 days)
TRADING_DAYS_PER_YEAR = 365

# Risk-free rate = 0% for a pure crypto portfolio.
# No T-bill baseline is meaningful when every asset is speculative.
RISK_FREE_RATE = 0.0

# Historical crash magnitude: 30% drop in one week.
# Validated by real events:
#   • BTC -35%  (May 2021 — China mining ban panic)
#   • LUNA -99% (June 2022 — algorithmic stablecoin collapse)
#   • FTX  -30% (November 2022 — exchange insolvency contagion)
CRASH_SCENARIO_PCT = 0.30

# yfinance appends this suffix to crypto symbols
YAHOO_SUFFIX = "-USD"

# Maps common user-typed symbols to valid yfinance ticker strings.
# Extend this dict freely for additional coins.
TICKER_ALIASES: dict[str, str] = {
    "BTC":   "BTC-USD",
    "ETH":   "ETH-USD",
    "BNB":   "BNB-USD",
    "SOL":   "SOL-USD",
    "ADA":   "ADA-USD",
    "XRP":   "XRP-USD",
    "DOT":   "DOT-USD",
    "DOGE":  "DOGE-USD",
    "AVAX":  "AVAX-USD",
    "MATIC": "MATIC-USD",
    "LINK":  "LINK-USD",
    "LTC":   "LTC-USD",
    "ATOM":  "ATOM-USD",
    "UNI":   "UNI-USD",
    "NEAR":  "NEAR-USD",
    "SHIB":  "SHIB-USD",
    "TRX":   "TRX-USD",
    "OP":    "OP-USD",
    "ARB":   "ARB-USD",
    "SUI":   "SUI-USD",
}


# ──────────────────────────────────────────────────────────────────
# SECTION B: DATA FETCHING
# ──────────────────────────────────────────────────────────────────
#
# WHY yf.Ticker().history() INSTEAD OF yf.download()?
#
# In yfinance >= 0.2.x, yf.download() returns MultiIndex columns even
# for a single ticker — e.g. ("Close", "BTC-USD") instead of "Close".
# This silently breaks any `"Close" in data.columns` check, causing every
# asset fetch to fail.
#
# yf.Ticker(ticker).history() is the stable, object-oriented API. It
# always returns a clean, flat-column DataFrame (Open, High, Low, Close,
# Volume) regardless of library version — the reliable choice for
# production scripts and Colab environments alike.


def resolve_ticker(symbol: str) -> str:
    """
    Converts a user-typed coin symbol to a yfinance-compatible ticker.

    Checks the TICKER_ALIASES map first. Falls back to auto-appending
    the '-USD' suffix for any coin not explicitly listed.

    Args:
        symbol (str): Raw user input, e.g. 'btc' or 'BTC'.

    Returns:
        str: yfinance ticker string, e.g. 'BTC-USD'.
    """
    symbol = symbol.strip().upper()
    if symbol in TICKER_ALIASES:
        return TICKER_ALIASES[symbol]
    return f"{symbol}{YAHOO_SUFFIX}"


def fetch_price_history(ticker: str, days: int = 90) -> "pd.Series | None":
    """
    Fetches historical daily closing prices via the Ticker object API.

    Args:
        ticker (str): Valid yfinance ticker, e.g. 'ETH-USD'.
        days (int):   Number of calendar days of history to fetch (default 90).

    Returns:
        pd.Series: Time-indexed daily closing prices, or None on any failure.
    """
    try:
        asset = yf.Ticker(ticker)
        data  = asset.history(period=f"{days}d")

        if data is None or data.empty:
            console.print(
                f"[dim red]   ↳ Error detail: .history() returned empty "
                f"DataFrame for '{ticker}'[/dim red]"
            )
            return None

        if "Close" not in data.columns:
            console.print(
                f"[dim red]   ↳ Error detail: 'Close' column missing. "
                f"Columns found: {list(data.columns)}[/dim red]"
            )
            return None

        close_series = data["Close"].dropna()

        if len(close_series) < 10:
            console.print(
                f"[dim red]   ↳ Error detail: Only {len(close_series)} data "
                f"points — need at least 10.[/dim red]"
            )
            return None

        return close_series

    except Exception as e:
        console.print(
            f"[dim red]   ↳ Error detail: {type(e).__name__}: {e}[/dim red]"
        )
        return None


def get_current_price(ticker: str) -> "float | None":
    """
    Retrieves the most recent daily closing price for a crypto asset.

    Fetches a short 5-day window and returns the last available value.

    Args:
        ticker (str): Valid yfinance ticker, e.g. 'BTC-USD'.

    Returns:
        float: Latest closing price in USD, or None on failure.
    """
    try:
        asset = yf.Ticker(ticker)
        data  = asset.history(period="5d")

        if data is None or data.empty or "Close" not in data.columns:
            return None

        close_series = data["Close"].dropna()
        if len(close_series) == 0:
            return None

        return float(close_series.iloc[-1])

    except Exception as e:
        console.print(
            f"[dim red]   ↳ get_current_price error for '{ticker}': "
            f"{type(e).__name__}: {e}[/dim red]"
        )
        return None


# ──────────────────────────────────────────────────────────────────
# SECTION C: RISK METRICS ENGINE
# ──────────────────────────────────────────────────────────────────


def compute_daily_returns(prices: pd.Series) -> pd.Series:
    """
    Computes daily logarithmic returns from a price series.

    WHY LOG RETURNS?
    Log returns — ln(P_t / P_{t-1}) — are preferred over simple percentage
    returns in quantitative finance for three reasons:
      1. Time-additive: the log return over N days equals the sum of N daily
         log returns, simplifying multi-period analysis.
      2. More symmetric and approximately normally distributed, which makes
         statistical tests (like Sharpe) more reliable.
      3. Mathematically prevent negative price values in simulations.

    Args:
        prices (pd.Series): Daily closing price series.

    Returns:
        pd.Series: Daily log-return series (length = len(prices) - 1).
    """
    return np.log(prices / prices.shift(1)).dropna()


def compute_annualized_volatility(returns: pd.Series) -> float:
    """
    Computes Annualized Volatility (standard deviation) of an asset.

    FINANCIAL LOGIC:
    Volatility is the standard deviation of daily log returns, scaled to
    an annual figure using the square-root-of-time rule:

        σ_annual = σ_daily × √(TRADING_DAYS_PER_YEAR)

    The square-root scaling follows from the assumption that daily returns
    are independent and identically distributed (i.i.d.) — so variance
    scales linearly with time, and standard deviation with √time.

    CONTEXT:
    Crypto assets typically exhibit 60–150% annualized volatility.
    Traditional equities (S&P 500) sit at roughly 15–20%.

    Args:
        returns (pd.Series): Daily log-return series.

    Returns:
        float: Annualized volatility as a decimal (e.g., 0.85 = 85%).
    """
    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def compute_annualized_return(returns: pd.Series) -> float:
    """
    Computes the Annualized Mean Return of an asset.

    FINANCIAL LOGIC:
    Compounds the average daily return over a full year:

        R_annual = (1 + μ_daily)^365 − 1

    Where μ_daily is the arithmetic mean of daily log returns.
    This converts a short observation window (e.g. 90 days) into a
    comparable annual figure for cross-asset benchmarking.

    Args:
        returns (pd.Series): Daily log-return series.

    Returns:
        float: Annualized return as a decimal (e.g., 0.50 = 50%).
    """
    mean_daily = returns.mean()
    return (1 + mean_daily) ** TRADING_DAYS_PER_YEAR - 1


def compute_sharpe_ratio(annualized_return: float, annualized_vol: float) -> float:
    """
    Computes the Sharpe Ratio for a single asset or a portfolio.

    FINANCIAL LOGIC — THE SHARPE RATIO:
    Developed by Nobel Laureate William Sharpe (1966), this is the gold-standard
    metric for risk-adjusted performance. It answers one question:
    "How much return am I earning per unit of risk I'm accepting?"

        Sharpe Ratio = (R_p − R_f) / σ_p

    Where:
        R_p = Annualized return of the asset/portfolio
        R_f = Risk-free rate (0% here — pure crypto comparison)
        σ_p = Annualized volatility

    INTERPRETATION GUIDE:
        < 0    → Negative risk-adjusted return (worse than holding cash)
        0–0.5  → Poor — not compensated adequately for the risk taken
        0.5–1.0 → Acceptable — moderate risk-adjusted performance
        1.0–2.0 → Good — solid risk compensation
        > 2.0  → Excellent (genuinely rare in crypto)

    Args:
        annualized_return (float): Annualized mean return.
        annualized_vol (float):    Annualized volatility.

    Returns:
        float: Sharpe Ratio. Returns 0.0 if volatility is zero (edge case guard).
    """
    if annualized_vol == 0:
        return 0.0
    return (annualized_return - RISK_FREE_RATE) / annualized_vol


def compute_portfolio_metrics(holdings: dict) -> "dict | None":
    """
    Master orchestrator: fetches live data and computes all risk metrics.

    PORTFOLIO MATHEMATICS:
        Portfolio Return    = Σ(w_i × R_i)      — weighted mean of asset returns
        Portfolio Volatility ≈ Σ(w_i × σ_i)    — weighted mean of asset vols
                                                   (conservative proxy; true value
                                                    accounting for correlations would
                                                    be lower for diversified portfolios)
        Portfolio Sharpe    = R_portfolio / σ_portfolio

    DEFENSIVE DESIGN — SEVEN NAMED GUARDS:
    Each guard validates one specific failure condition. An asset failing
    any guard is appended to failed_assets and excluded from all math.
    This prevents TypeError / ZeroDivisionError / NaN propagation from
    ever reaching the calculation layer.

        Guard 1 — None from fetch_price_history (network / rate-limit / bad ticker)
        Guard 2 — Non-Series or empty Series
        Guard 3 — Fewer than 2 rows (cannot compute even one daily return)
        Guard 4 — None or zero current price from get_current_price
        Guard 5 — Total portfolio value ≤ 0 (division-by-zero on weights)
        Guard 6 — All-NaN or all-Inf daily return series
        Guard 7 — try/except around the volatility / Sharpe computation block

    Args:
        holdings (dict): { "BTC": 0.5, "ETH": 2.0, ... }

    Returns:
        dict: Full results dictionary, or None if zero assets survive all guards.
    """
    prices:          dict[str, float]      = {}
    price_histories: dict[str, pd.Series]  = {}
    failed_assets:   list[str]             = []

    # ── Step 1: Fetch & validate data for every asset ────────────
    console.print(
        "\n[bold cyan]📡 Fetching live market data from Yahoo Finance...[/bold cyan]"
    )

    for symbol, qty in holdings.items():
        ticker = resolve_ticker(symbol)
        console.print(
            f"   [dim]→ Fetching [yellow]{symbol}[/yellow] ({ticker})...[/dim]",
            end="",
        )

        history = fetch_price_history(ticker, days=90)

        # Guard 1: None return
        if history is None:
            console.print(" [red]✗ No data returned. Skipping.[/red]")
            failed_assets.append(symbol)
            continue

        # Guard 2: Must be a non-empty pd.Series
        if not isinstance(history, pd.Series) or history.empty:
            console.print(" [red]✗ Empty price series. Skipping.[/red]")
            failed_assets.append(symbol)
            continue

        # Guard 3: Need ≥ 2 rows to compute one daily return
        if len(history) < 2:
            console.print(
                f" [red]✗ Insufficient history ({len(history)} row). Skipping.[/red]"
            )
            failed_assets.append(symbol)
            continue

        price = get_current_price(ticker)

        # Guard 4: No usable current price
        if price is None or price <= 0:
            console.print(" [red]✗ Could not resolve current price. Skipping.[/red]")
            failed_assets.append(symbol)
            continue

        prices[symbol]          = price
        price_histories[symbol] = history
        console.print(
            f" [green]✓ ${price:,.4f}  ({len(history)} days of data)[/green]"
        )

    # ── Step 2: Bail out cleanly if nothing is usable ────────────
    valid_holdings = {k: v for k, v in holdings.items() if k not in failed_assets}

    if not valid_holdings:
        return None

    # ── Step 3: Portfolio value & weights ────────────────────────
    asset_values = {
        symbol: qty * prices[symbol]
        for symbol, qty in valid_holdings.items()
    }
    total_portfolio_value = sum(asset_values.values())

    # Guard 5: Prevent division-by-zero when computing weights
    if total_portfolio_value <= 0:
        console.print(
            "[bold red]✗ Total portfolio value resolved to zero. "
            "Cannot proceed.[/bold red]"
        )
        return None

    weights = {
        symbol: value / total_portfolio_value
        for symbol, value in asset_values.items()
    }

    # ── Step 4: Per-asset risk metrics ───────────────────────────
    asset_metrics: dict = {}

    for symbol in valid_holdings:
        history = price_histories[symbol]

        try:
            returns = compute_daily_returns(history)

            # Guard 6a: Returns Series must be non-empty and contain real values
            if returns is None or returns.empty or returns.isna().all():
                console.print(
                    f"[yellow]⚠  {symbol}: return series is all NaN "
                    f"— skipping metrics.[/yellow]"
                )
                failed_assets.append(symbol)
                continue

            # Scrub any residual NaN / Inf that would corrupt std() and mean()
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

            # Guard 6b: Need ≥ 2 clean returns for std() to be meaningful
            if len(returns) < 2:
                console.print(
                    f"[yellow]⚠  {symbol}: fewer than 2 clean returns "
                    f"— skipping metrics.[/yellow]"
                )
                failed_assets.append(symbol)
                continue

            ann_vol = compute_annualized_volatility(returns)
            ann_ret = compute_annualized_return(returns)
            sharpe  = compute_sharpe_ratio(ann_ret, ann_vol)

        except Exception as e:
            # Guard 7: Catch unexpected math failures without crashing the app
            console.print(
                f"[yellow]⚠  {symbol}: metric calculation failed — "
                f"{type(e).__name__}: {e}. Skipping.[/yellow]"
            )
            failed_assets.append(symbol)
            continue

        asset_metrics[symbol] = {
            "price":         prices[symbol],
            "quantity":      valid_holdings[symbol],
            "value_usd":     asset_values[symbol],
            "weight":        weights[symbol],
            "ann_return":    ann_ret,
            "ann_vol":       ann_vol,
            "sharpe":        sharpe,
            "daily_returns": returns,
        }

    # ── Step 5: Re-check after metric filtering ───────────────────
    if not asset_metrics:
        return None

    # Recompute weights using only assets that survived all guards
    surviving_value = sum(m["value_usd"] for m in asset_metrics.values())
    for symbol in asset_metrics:
        asset_metrics[symbol]["weight"] = (
            asset_metrics[symbol]["value_usd"] / surviving_value
        )

    # ── Step 6: Portfolio-level aggregates ───────────────────────
    port_return = sum(
        m["weight"] * m["ann_return"] for m in asset_metrics.values()
    )
    port_vol = sum(
        m["weight"] * m["ann_vol"] for m in asset_metrics.values()
    )
    port_sharpe = compute_sharpe_ratio(port_return, port_vol)

    return {
        "asset_metrics":        asset_metrics,
        "total_value_usd":      surviving_value,
        "portfolio_return":     port_return,
        "portfolio_volatility": port_vol,
        "portfolio_sharpe":     port_sharpe,
        "failed_assets":        failed_assets,
    }


# ──────────────────────────────────────────────────────────────────
# SECTION D: HISTORICAL CRASH SCENARIO
# ──────────────────────────────────────────────────────────────────


def run_crash_scenario(results: dict) -> dict:
    """
    Simulates a 'Black Week' historical crypto market crash scenario.

    FINANCIAL LOGIC — DETERMINISTIC STRESS TEST:
    Rather than a probabilistic Monte Carlo simulation, we apply a fixed
    30% instantaneous shock to every asset simultaneously. This mirrors
    actual crypto market behaviour during systemic crises, where high
    cross-asset correlation means nearly all coins fall together.

    Formula:
        Loss            = Portfolio_Value × CRASH_SCENARIO_PCT
        Surviving_Value = Portfolio_Value × (1 − CRASH_SCENARIO_PCT)

    A deterministic stress test is deliberately conservative and easy to
    interpret — ideal for explaining risk to a non-technical audience.

    Args:
        results (dict): Output from compute_portfolio_metrics().

    Returns:
        dict: Pre-crash value, total loss, post-crash value, and a per-asset
              breakdown of the same figures.
    """
    total_value     = results["total_value_usd"]
    loss_amount     = total_value * CRASH_SCENARIO_PCT
    surviving_value = total_value * (1 - CRASH_SCENARIO_PCT)

    asset_crash: dict = {}
    for symbol, metrics in results["asset_metrics"].items():
        asset_crash[symbol] = {
            "pre_crash_value":  metrics["value_usd"],
            "loss":             metrics["value_usd"] * CRASH_SCENARIO_PCT,
            "post_crash_value": metrics["value_usd"] * (1 - CRASH_SCENARIO_PCT),
        }

    return {
        "crash_pct":       CRASH_SCENARIO_PCT,
        "total_pre":       total_value,
        "total_loss":      loss_amount,
        "total_post":      surviving_value,
        "asset_breakdown": asset_crash,
    }


# ──────────────────────────────────────────────────────────────────
# SECTION E: RISK PROFILING ENGINE
# ──────────────────────────────────────────────────────────────────


def classify_risk_profile(portfolio_sharpe: float, portfolio_vol: float) -> dict:
    """
    Classifies the portfolio into one of three risk profile tiers.

    TWO-AXIS SCORING SYSTEM:

    Axis 1 — SHARPE SCORE (quality of risk-adjusted returns):
        > 1.0   → +2 pts   Great: well-compensated for the risk taken
        0–1.0   → +1 pt    Average: some positive compensation
        < 0     → +0 pts   Poor: losing money on a risk-adjusted basis

    Axis 2 — VOLATILITY SCORE (magnitude of price risk):
        < 60%   → +2 pts   Relatively calm by crypto standards
        60–100% → +1 pt    Typical crypto range
        > 100%  → +0 pts   Extreme — deep altcoin / meme-coin territory

    TOTAL SCORE → PROFILE TIER:
        3–4 pts → Conservative    (disciplined, balanced portfolio)
        2 pts   → Moderate        (willing risk-takers, manageable exposure)
        0–1 pts → Aggressive Degen (speculative, high drawdown risk)

    Args:
        portfolio_sharpe (float): Portfolio-level Sharpe Ratio.
        portfolio_vol (float):    Portfolio-level annualized volatility.

    Returns:
        dict: Profile name, score, Rich color, emoji, description text,
              and a boolean flag indicating whether rebalancing is needed.
    """
    score = 0

    # Sharpe axis
    if portfolio_sharpe > 1.0:
        score += 2
    elif portfolio_sharpe >= 0:
        score += 1

    # Volatility axis
    if portfolio_vol < 0.60:
        score += 2
    elif portfolio_vol <= 1.00:
        score += 1

    if score >= 3:
        return {
            "name":            "Conservative",
            "score":           score,
            "color":           "green",
            "emoji":           "🛡️",
            "description":     (
                "Your portfolio shows disciplined risk management. "
                "Lower volatility and decent risk-adjusted returns suggest "
                "a balanced, sustainable long-term strategy."
            ),
            "needs_rebalance": False,
        }
    elif score == 2:
        return {
            "name":            "Moderate",
            "score":           score,
            "color":           "yellow",
            "emoji":           "⚖️",
            "description":     (
                "Your portfolio takes on meaningful crypto risk but isn't "
                "at the extreme end. You're comfortable with price swings "
                "but should monitor concentration risk closely."
            ),
            "needs_rebalance": False,
        }
    else:
        return {
            "name":            "Aggressive Degen",
            "score":           score,
            "color":           "red",
            "emoji":           "🎰",
            "description":     (
                "High volatility with poor risk-adjusted returns signals "
                "a highly speculative portfolio. You are exposed to severe "
                "drawdown risk. Immediate rebalancing is strongly advised."
            ),
            "needs_rebalance": True,
        }


def generate_rebalancing_plan(results: dict, rebalance_pct: float = 0.15) -> dict:
    """
    Generates a concrete stablecoin rebalancing plan for Aggressive Degen portfolios.

    FINANCIAL LOGIC — WHY STABLECOINS?
    Stablecoins (USDT, USDC) maintain a 1:1 peg with the US Dollar, providing:
      1. Zero price volatility — a direct shield against market swings.
      2. Instant liquidity — deployable immediately when crash opportunities arise.
      3. Optionality — capital ready to 'buy the dip' at market bottoms.

    By shifting 15% of a volatile portfolio into stablecoins, the effective
    portfolio volatility drops by approximately 15% immediately, due to the
    weighted-average property of portfolio volatility:

        σ_new ≈ (0.85 × σ_crypto) + (0.15 × 0%)  →  0.85 × σ_crypto

    SELL PLAN (proportional reduction):
        Each crypto asset contributes proportionally to the rebalancing fund.
        sell_amount_i = asset_value_i × rebalance_pct
        new_value_i   = asset_value_i − sell_amount_i

    Args:
        results (dict):        Output from compute_portfolio_metrics().
        rebalance_pct (float): Fraction to redirect into stablecoins (default 15%).

    Returns:
        dict: Complete rebalancing plan with per-asset sell quantities (USD and
              coin units), remaining values, new weights, and total USDT to buy.
    """
    total_value = results["total_value_usd"]
    usdt_target = total_value * rebalance_pct
    remaining   = total_value * (1 - rebalance_pct)

    sell_plan: dict = {}
    for symbol, metrics in results["asset_metrics"].items():
        sell_amount        = metrics["value_usd"] * rebalance_pct
        new_value          = metrics["value_usd"] - sell_amount
        sell_plan[symbol]  = {
            "current_value": metrics["value_usd"],
            "sell_usd":      sell_amount,
            "sell_qty":      sell_amount / metrics["price"],
            "new_value":     new_value,
            "new_weight":    new_value / total_value,
        }

    return {
        "rebalance_pct":    rebalance_pct,
        "usdt_to_buy":      usdt_target,
        "per_asset_sells":  sell_plan,
        "new_crypto_value": remaining,
        "new_usdt_value":   usdt_target,
    }


# ──────────────────────────────────────────────────────────────────
# SECTION F: RICH UI RENDERING FUNCTIONS
# ──────────────────────────────────────────────────────────────────


def render_banner() -> None:
    """Renders the application title banner to the terminal."""
    title = Text()
    title.append("₿  CRYPTO RISK PROFILER TERMINAL  ₿\n",   style="bold white on dark_blue")
    title.append("   UOB My Digital Space CODEFEST 2026   \n", style="bold cyan")
    title.append(" Financial Literacy · Quantitative Risk Engine ", style="italic dim white")

    console.print(Panel(
        Align.center(title),
        border_style="bright_blue",
        padding=(1, 4),
    ))
    console.print()


def render_portfolio_table(results: dict) -> None:
    """
    Renders a colour-coded portfolio allocation table.

    Colour coding logic (applied independently per metric):
        Returns:    green = positive  |  red = negative
        Volatility: green < 60%  |  yellow 60–100%  |  red > 100%
        Sharpe:     green > 1.0  |  yellow 0–1.0    |  red < 0
    """
    table = Table(
        title="📊 Portfolio Composition & Risk Metrics",
        box=box.DOUBLE_EDGE,
        border_style="bright_blue",
        header_style="bold white on navy_blue",
        show_lines=True,
    )

    table.add_column("Asset",             style="bold yellow", justify="center", min_width=8)
    table.add_column("Qty Held",          style="cyan",        justify="right",  min_width=12)
    table.add_column("Price (USD)",       style="white",       justify="right",  min_width=14)
    table.add_column("Value (USD)",       style="bold green",  justify="right",  min_width=16)
    table.add_column("Allocation %",      style="magenta",     justify="center", min_width=12)
    table.add_column("Ann. Return",                            justify="center", min_width=13)
    table.add_column("Ann. Volatility",                        justify="center", min_width=14)
    table.add_column("Sharpe Ratio",                           justify="center", min_width=13)

    for symbol, m in results["asset_metrics"].items():
        # Return colour
        ret_color = "green" if m["ann_return"] >= 0 else "red"
        ret_txt   = f"[{ret_color}]{m['ann_return']*100:+.1f}%[/{ret_color}]"

        # Volatility colour
        vol_color = "green" if m["ann_vol"] < 0.60 else ("yellow" if m["ann_vol"] <= 1.00 else "red")
        vol_txt   = f"[{vol_color}]{m['ann_vol']*100:.1f}%[/{vol_color}]"

        # Sharpe colour
        sh_color  = "green" if m["sharpe"] > 1.0 else ("yellow" if m["sharpe"] >= 0 else "red")
        sh_txt    = f"[{sh_color}]{m['sharpe']:.3f}[/{sh_color}]"

        table.add_row(
            f"[bold yellow]{symbol}[/bold yellow]",
            f"{m['quantity']:,.6f}",
            f"${m['price']:>12,.4f}",
            f"${m['value_usd']:>14,.2f}",
            f"{m['weight']*100:.1f}%",
            ret_txt,
            vol_txt,
            sh_txt,
        )

    # Portfolio summary footer row
    p_r  = results["portfolio_return"]
    p_v  = results["portfolio_volatility"]
    p_s  = results["portfolio_sharpe"]

    table.add_section()
    table.add_row(
        "[bold white]PORTFOLIO[/bold white]",
        "—", "—",
        f"[bold green]${results['total_value_usd']:>14,.2f}[/bold green]",
        "[bold]100.0%[/bold]",
        f"[{'green' if p_r >= 0 else 'red'}]{p_r*100:+.1f}%[/]",
        f"[{'yellow' if p_v <= 1.0 else 'red'}]{p_v*100:.1f}%[/]",
        f"[{'green' if p_s > 1 else 'yellow' if p_s >= 0 else 'red'}]{p_s:.3f}[/]",
    )

    console.print(table)


def render_crash_table(crash: dict) -> None:
    """Renders the historical crash stress-test results table."""
    table = Table(
        title=(
            f"⚠️  Historical Crash Scenario: "
            f"-{crash['crash_pct']*100:.0f}% Market Drop in 1 Week"
        ),
        box=box.SIMPLE_HEAD,
        border_style="dark_orange",
        header_style="bold white on dark_red",
        show_lines=True,
    )

    table.add_column("Asset",            style="yellow",    justify="center", min_width=8)
    table.add_column("Pre-Crash Value",  style="white",     justify="right",  min_width=16)
    table.add_column("Loss (USD)",       style="bold red",  justify="right",  min_width=16)
    table.add_column("Post-Crash Value", style="dim white", justify="right",  min_width=16)

    for symbol, ac in crash["asset_breakdown"].items():
        table.add_row(
            f"[bold yellow]{symbol}[/bold yellow]",
            f"${ac['pre_crash_value']:>14,.2f}",
            f"[red]-${ac['loss']:>13,.2f}[/red]",
            f"${ac['post_crash_value']:>14,.2f}",
        )

    table.add_section()
    table.add_row(
        "[bold white]TOTAL[/bold white]",
        f"[bold]${crash['total_pre']:>14,.2f}[/bold]",
        f"[bold red]-${crash['total_loss']:>13,.2f}[/bold red]",
        f"[bold]${crash['total_post']:>14,.2f}[/bold]",
    )

    console.print(table)


def render_risk_profile(profile: dict, results: dict) -> None:
    """Renders the risk profile verdict inside a colour-bordered Rich panel."""
    color = profile["color"]
    emoji = profile["emoji"]
    name  = profile["name"]
    desc  = profile["description"]
    score = profile["score"]

    content = Text()
    content.append(f"\n  {emoji}  RISK PROFILE: ",  style=f"bold {color}")
    content.append(f"{name.upper()}\n\n",            style=f"bold {color} underline")
    content.append(f"  Risk Score: {score}/4\n\n",   style="dim white")
    content.append("  📈 Annualized Return:     ",   style="white")
    content.append(f"{results['portfolio_return']*100:+.2f}%\n",    style="bold cyan")
    content.append("  📉 Annualized Volatility: ",   style="white")
    content.append(f"{results['portfolio_volatility']*100:.2f}%\n", style="bold magenta")
    content.append("  ⚡ Sharpe Ratio:          ",   style="white")
    content.append(f"{results['portfolio_sharpe']:.4f}\n\n",        style="bold yellow")
    content.append(f"  {desc}\n",                    style="italic white")

    console.print(Panel(
        content,
        title=f"[bold {color}]▌ RISK VERDICT ▐[/bold {color}]",
        border_style=color,
        padding=(0, 2),
    ))


def render_rebalancing_plan(plan: dict) -> None:
    """Renders the stablecoin rebalancing recommendation and per-asset sell table."""
    console.print(Panel(
        f"[bold red]⚠  ACTION REQUIRED:[/bold red] [white]Your portfolio is classified as "
        f"[bold red]Aggressive Degen[/bold red].\n"
        f"   Recommendation: Shift [bold yellow]{plan['rebalance_pct']*100:.0f}%[/bold yellow] "
        f"of holdings into [bold green]USDT/USDC stablecoins[/bold green] to reduce volatility "
        f"exposure.\n"
        f"   This reduces effective portfolio volatility by "
        f"~{plan['rebalance_pct']*100:.0f}% immediately.[/white]",
        title="[bold red]🔄 REBALANCING RECOMMENDATION[/bold red]",
        border_style="red",
        padding=(1, 2),
    ))

    table = Table(
        title="Suggested Sell Orders to Fund Stablecoin Position",
        box=box.ROUNDED,
        border_style="yellow",
        header_style="bold white on dark_orange3",
        show_lines=True,
    )

    table.add_column("Asset",         style="yellow",    justify="center", min_width=8)
    table.add_column("Current Value", style="white",     justify="right",  min_width=16)
    table.add_column("Sell (USD)",    style="bold red",  justify="right",  min_width=14)
    table.add_column("Sell (Qty)",    style="red",       justify="right",  min_width=14)
    table.add_column("Remaining",     style="dim green", justify="right",  min_width=16)
    table.add_column("New Weight",    style="magenta",   justify="center", min_width=11)

    for symbol, sell in plan["per_asset_sells"].items():
        table.add_row(
            f"[bold yellow]{symbol}[/bold yellow]",
            f"${sell['current_value']:>13,.2f}",
            f"[red]-${sell['sell_usd']:>11,.2f}[/red]",
            f"[red]{sell['sell_qty']:>12,.6f}[/red]",
            f"${sell['new_value']:>13,.2f}",
            f"{sell['new_weight']*100:.1f}%",
        )

    console.print(table)
    console.print(
        f"\n  ➡  [bold green]Buy ${plan['usdt_to_buy']:,.2f} worth of "
        f"USDT or USDC[/bold green] after executing the sells above."
    )
    console.print(
        f"  ➡  New crypto exposure: [cyan]${plan['new_crypto_value']:,.2f}[/cyan]  |  "
        f"Stablecoin buffer: [green]${plan['new_usdt_value']:,.2f}[/green]\n"
    )


# ──────────────────────────────────────────────────────────────────
# SECTION G: INTERACTIVE MAIN LOOP
# ──────────────────────────────────────────────────────────────────


def run_crypto_risk_profiler() -> None:
    """
    Entry point — orchestrates the full interactive CLI session.

    Interaction design:
        • All input is collected via Python's built-in input() inside
          while loops — no click, argparse, or sys.argv required.
        • This keeps the program compatible with standard terminals,
          VS Code integrated terminals, and Google Colab alike.

    Session flow:
        1.  Render banner
        2.  Portfolio input loop  (type symbol + quantity, 'done' to finish)
        3.  Fetch live prices & compute all risk metrics (with full error handling)
        4.  Render portfolio allocation table
        5.  Run 30% historical crash scenario & render table
        6.  Classify risk profile (Conservative / Moderate / Aggressive Degen)
        7.  Render rebalancing plan if profile is Aggressive Degen
        8.  Prompt to analyse another portfolio or exit
    """
    render_banner()

    while True:

        # ── STEP 1: Portfolio Input ───────────────────────────────
        console.print(Rule("[bold cyan]💼 PORTFOLIO INPUT[/bold cyan]", style="bright_blue"))
        console.print(
            "[dim]Enter each crypto asset and the quantity you hold.\n"
            "Type [bold]'done'[/bold] when finished. "
            "Type [bold]'exit'[/bold] to quit the app.[/dim]\n"
        )

        holdings: dict[str, float] = {}

        while True:
            raw = input("  🪙 Enter coin symbol (e.g. BTC, ETH, SOL) or 'done': ").strip()

            if raw.lower() == "exit":
                console.print(
                    "\n[bold cyan]👋 Thanks for using Crypto Risk Profiler. "
                    "Stay SAFU![/bold cyan]\n"
                )
                return

            if raw.lower() == "done":
                if not holdings:
                    console.print(
                        "  [yellow]⚠  Please enter at least one asset first.[/yellow]"
                    )
                    continue
                break

            symbol = raw.upper()

            if not symbol.isalpha() or len(symbol) > 10:
                console.print(
                    "  [red]✗ Invalid symbol. Use letters only (e.g. BTC, ETH, SOL).[/red]"
                )
                continue

            if symbol in holdings:
                console.print(
                    f"  [yellow]⚠  {symbol} is already in your portfolio. "
                    f"Enter a different coin.[/yellow]"
                )
                continue

            # Inner loop: validate quantity
            while True:
                qty_raw = input(f"  📦 How many {symbol} do you hold? ").strip()
                try:
                    qty = float(qty_raw)
                    if qty <= 0:
                        console.print(
                            "  [red]✗ Quantity must be greater than 0.[/red]"
                        )
                        continue
                    break
                except ValueError:
                    console.print(
                        "  [red]✗ Please enter a valid number "
                        "(e.g. 0.5, 10, 1000).[/red]"
                    )

            holdings[symbol] = qty
            console.print(f"  [green]✓ Added: {qty:,.6f} × {symbol}[/green]")

        # ── STEP 2: Compute Metrics — full defensive wrapper ──────
        console.print()
        console.print(
            Rule("[bold cyan]⚙️  COMPUTING RISK METRICS[/bold cyan]", style="bright_blue")
        )

        try:
            results = compute_portfolio_metrics(holdings)

        except Exception as e:
            # Last-resort catch: something unexpected escaped all inner guards
            console.print(Panel(
                f"[bold red]Unexpected error during analysis:[/bold red]\n"
                f"[dim]{type(e).__name__}: {e}[/dim]\n\n"
                f"[yellow]Please try again. If this persists, check your "
                f"internet connection.[/yellow]",
                title="[bold red]💥 FATAL ERROR[/bold red]",
                border_style="red",
            ))
            continue  # Return to portfolio input — do not crash the process

        # All assets failed their data-quality guards
        if results is None:
            console.print(Panel(
                "[bold red]Error:[/bold red] Failed to fetch market data from the server.\n\n"
                "[white]This is usually caused by:[/white]\n"
                "  [yellow]•[/yellow] Yahoo Finance rate-limiting your IP address\n"
                "  [yellow]•[/yellow] Invalid or unlisted coin symbols\n"
                "  [yellow]•[/yellow] A temporary network outage\n\n"
                "[bold cyan]What to try:[/bold cyan]\n"
                "  [green]1.[/green] Wait 30–60 seconds and try again\n"
                "  [green]2.[/green] Double-check your coin symbols (e.g. BTC, ETH, SOL)\n"
                "  [green]3.[/green] Try a smaller portfolio (2–3 coins) to isolate the issue",
                title="[bold red]⚠  DATA FETCH FAILED[/bold red]",
                border_style="dark_orange",
                padding=(1, 2),
            ))
            continue

        # Partial success — surface any skipped assets to the user
        if results["failed_assets"]:
            console.print(
                f"\n[yellow]⚠  Skipped assets (data unavailable): "
                f"[bold]{', '.join(results['failed_assets'])}[/bold][/yellow]\n"
            )

        # All fetched assets then failed metric-calculation guards
        if not results["asset_metrics"]:
            console.print(Panel(
                "[bold red]Error:[/bold red] All fetched assets failed "
                "during metric calculation.\n"
                "[white]This can happen when price data is too sparse or entirely NaN.\n"
                "Please wait a moment and try again with different assets.[/white]",
                title="[bold red]⚠  NO VALID ASSETS[/bold red]",
                border_style="red",
            ))
            continue

        # ── STEP 3: Portfolio Table ───────────────────────────────
        console.print()
        console.print(
            Rule("[bold cyan]📊 ANALYSIS RESULTS[/bold cyan]", style="bright_blue")
        )
        console.print()
        render_portfolio_table(results)

        # ── STEP 4: Historical Crash Scenario ─────────────────────
        console.print()
        crash_results = run_crash_scenario(results)
        render_crash_table(crash_results)

        # ── STEP 5: Risk Profile Classification ───────────────────
        console.print()
        profile = classify_risk_profile(
            results["portfolio_sharpe"],
            results["portfolio_volatility"],
        )
        render_risk_profile(profile, results)

        # ── STEP 6: Rebalancing Plan (Aggressive Degen only) ──────
        if profile["needs_rebalance"]:
            console.print()
            console.print(
                Rule("[bold red]🔄 REBALANCING PLAN[/bold red]", style="red")
            )
            plan = generate_rebalancing_plan(results, rebalance_pct=0.15)
            render_rebalancing_plan(plan)

        # ── STEP 7: Continue or Exit ──────────────────────────────
        console.print(Rule(style="bright_blue"))
        again = input("\n  🔁 Analyse another portfolio? (yes / no): ").strip().lower()

        if again not in ("yes", "y"):
            console.print(
                "\n[bold cyan]✅ Analysis complete. "
                "Remember: Not Financial Advice. "
                "Do Your Own Research. Stay SAFU! 🛡️[/bold cyan]\n"
            )
            break

        console.print("\n" + "═" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_crypto_risk_profiler()
