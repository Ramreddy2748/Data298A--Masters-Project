"""
critic_agent.py

Critic / Orchestrator Agent
---------------------------
Runs the three project agents for one ticker, prints every agent output,
and produces one final risk decision.

The critic can optionally use an OpenAI-compatible LLM endpoint for the final
reasoning step. If no critic API key is configured, it uses a transparent
weighted rule-based decision so the pipeline still runs locally.

Example:
    python agents/critic_agent.py A
    python agents/critic_agent.py A --company "Agilent Technologies" --sector Healthcare
    python agents/critic_agent.py A --use-llm
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SILVER_DIR = PROJECT_ROOT / "data" / "silver"
GOLD_DIR = PROJECT_ROOT / "data" / "gold"


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except Exception:
        return default


def _label_from_score(score: float) -> str:
    if score <= 3.5:
        return "LOW"
    if score <= 6.0:
        return "MODERATE"
    return "HIGH"


def _clamp_score(score: float) -> float:
    return round(max(1.0, min(10.0, float(score))), 2)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _latest_row(rows: list[dict[str, str]], date_columns: tuple[str, ...]) -> dict[str, str]:
    if not rows:
        return {}
    for col in date_columns:
        dated = [r for r in rows if r.get(col)]
        if dated:
            return sorted(dated, key=lambda r: str(r.get(col, "")))[-1]
    return rows[-1]


def _usd_m(value: Any) -> float:
    """Normalize values that may be raw USD or already USD millions."""
    number = _safe_float(value, 0.0) or 0.0
    if abs(number) > 1_000_000:
        return number / 1_000_000.0
    return number


def _load_gold_context(ticker: str) -> dict[str, str]:
    rows = _read_csv(GOLD_DIR / "gold_risk_scores_ALL.csv")
    ticker = ticker.upper()
    for row in rows:
        if str(row.get("ticker", "")).upper() == ticker:
            return row
    return {}


def run_fundamental_agent(ticker: str, company_name: Optional[str] = None) -> dict[str, Any]:
    """Rule implementation matching the fundamental fine-tune target logic."""
    ticker = ticker.upper()
    rows = _read_csv(SILVER_DIR / f"silver_edgar_{ticker}.csv")
    row = _latest_row(rows, ("end_date", "filed"))

    if not row:
        return {
            "agent": "Fundamental Agent",
            "ticker": ticker,
            "risk_score": 5.0,
            "risk_label": "MODERATE",
            "confidence": 0.15,
            "evidence": [f"No EDGAR silver file found for {ticker}."],
            "positive_signals": [],
            "negative_signals": ["Fundamental data unavailable."],
            "claim_type": "FALLBACK_ESTIMATE",
        }

    cash = _usd_m(row.get("cash_usd_m"))
    debt = _usd_m(row.get("long_term_debt"))
    net_income = _usd_m(row.get("net_income_usd_m"))
    operating_income = _usd_m(row.get("operating_income_usd_m"))
    revenue = _usd_m(row.get("revenue") or row.get("revenue_alt"))
    equity = _usd_m(row.get("stockholders_equity_usd_m") or row.get("stockholders_equity"))

    debt_to_cash = debt / (cash + 1.0)
    debt_to_equity = debt / (equity + 1.0) if equity > 0 else 3.0
    profit_margin = net_income / (revenue + 1.0) if revenue > 0 else 0.0
    operating_margin = operating_income / (revenue + 1.0) if revenue > 0 else 0.0

    score = (
        min(debt_to_cash / 5.0, 1.0) * 2.5
        + min(debt_to_equity / 2.0, 1.0) * 2.5
        + (1.0 - max(min(profit_margin / 0.20, 1.0), -1.0)) * 2.5
        + (1.0 - max(min(operating_margin / 0.25, 1.0), -1.0)) * 2.5
    )
    score = _clamp_score(score)

    positive, negative = [], []
    if debt_to_cash <= 2:
        positive.append("Cash reserves are reasonable relative to long-term debt.")
    else:
        negative.append("Long-term debt is elevated relative to cash reserves.")

    if debt_to_equity <= 1:
        positive.append("Debt-to-equity does not indicate severe leverage pressure.")
    elif debt_to_equity <= 2:
        negative.append("Debt-to-equity indicates moderate leverage exposure.")
    else:
        negative.append("Debt-to-equity indicates high leverage pressure.")

    if profit_margin >= 0.08:
        positive.append("Profit margin indicates positive earnings strength.")
    elif profit_margin >= 0:
        negative.append("Profit margin is low, limiting financial flexibility.")
    else:
        negative.append("Negative profit margin indicates profitability stress.")

    if operating_margin >= 0.10:
        positive.append("Operating margin shows stable core business performance.")
    elif operating_margin >= 0:
        negative.append("Operating margin is thin, suggesting limited operational cushion.")
    else:
        negative.append("Negative operating margin indicates weak core operations.")

    return {
        "agent": "Fundamental Agent",
        "ticker": ticker,
        "company": company_name or row.get("company") or ticker,
        "as_of_date": row.get("end_date") or row.get("filed"),
        "risk_score": score,
        "risk_label": _label_from_score(score),
        "confidence": 0.78 if rows else 0.15,
        "claim_type": "DERIVED_ESTIMATE",
        "metrics": {
            "cash_usd_m": round(cash, 2),
            "long_term_debt_usd_m": round(debt, 2),
            "revenue_usd_m": round(revenue, 2),
            "profit_margin": round(profit_margin, 4),
            "operating_margin": round(operating_margin, 4),
            "debt_to_cash": round(debt_to_cash, 4),
            "debt_to_equity": round(debt_to_equity, 4),
        },
        "evidence": [
            f"Cash {cash:.2f} USDm versus long-term debt {debt:.2f} USDm.",
            f"Profit margin {profit_margin:.2%}; operating margin {operating_margin:.2%}.",
            f"Debt-to-equity {debt_to_equity:.2f}.",
        ],
        "positive_signals": positive,
        "negative_signals": negative,
        "overall_assessment": (
            f"Fundamental risk is {_label_from_score(score)} based on leverage, "
            "profitability, and operating margin."
        ),
    }


def run_market_sentiment_agent(ticker: str, company_name: Optional[str] = None) -> dict[str, Any]:
    """Market + volatility + news sentiment agent."""
    ticker = ticker.upper()
    price_rows = _read_csv(SILVER_DIR / f"silver_prices_{ticker}.csv")
    price_row = _latest_row(price_rows, ("date",))
    news_rows = [
        r for r in _read_csv(SILVER_DIR / "silver_news_sentiment.csv")
        if str(r.get("ticker", "")).upper() == ticker
    ]

    if not price_row:
        return {
            "agent": "Market + Volatility + News Sentiment Agent",
            "ticker": ticker,
            "risk_score": 5.0,
            "risk_label": "MODERATE",
            "confidence": 0.15,
            "evidence": [f"No price silver file found for {ticker}."],
            "positive_signals": [],
            "negative_signals": ["Market data unavailable."],
            "claim_type": "FALLBACK_ESTIMATE",
        }

    daily_returns = [
        x for x in (_safe_float(r.get("daily_return")) for r in price_rows)
        if x is not None
    ]
    last_30_returns = daily_returns[-30:]
    realized_vol_30d = statistics.pstdev(last_30_returns) if len(last_30_returns) >= 2 else 0.0

    rolling_vol_30d = _safe_float(price_row.get("rolling_vol_30d"), realized_vol_30d) or 0.0
    rolling_vol_60d = _safe_float(price_row.get("rolling_vol_60d"), rolling_vol_30d) or rolling_vol_30d
    beta = _safe_float(price_row.get("beta"), 1.0) or 1.0
    price_vs_high = _safe_float(price_row.get("price_vs_52w_high_pct"), 0.0) or 0.0
    ytd_return = _safe_float(_load_gold_context(ticker).get("ytd_return_pct"), 0.0) or 0.0

    sentiment_values = [
        x for x in (_safe_float(r.get("sentiment_mean")) for r in news_rows)
        if x is not None
    ]
    avg_sentiment = statistics.mean(sentiment_values[-10:]) if sentiment_values else 0.0
    article_count = sum(int(_safe_float(r.get("article_count"), 0) or 0) for r in news_rows[-10:])

    # In this project some silver files store rolling volatility as an
    # annualized decimal (for example 0.26 = 26%), while raw daily-return
    # volatility is usually much smaller. Handle both forms.
    annual_vol_pct = (
        rolling_vol_30d * 100
        if rolling_vol_30d > 0.10
        else rolling_vol_30d * math.sqrt(252) * 100
    )
    vol_component = min(annual_vol_pct / 50.0, 1.0) * 3.0
    beta_component = min(max(beta - 0.8, 0.0) / 1.2, 1.0) * 2.0
    drawdown_component = min(abs(min(price_vs_high, 0.0)) / 40.0, 1.0) * 2.0
    sentiment_component = min(max(-avg_sentiment, 0.0) / 0.6, 1.0) * 2.0
    ytd_component = min(abs(min(ytd_return, 0.0)) / 35.0, 1.0) * 1.0
    score = _clamp_score(1.0 + vol_component + beta_component + drawdown_component + sentiment_component + ytd_component)

    positive, negative = [], []
    if rolling_vol_30d <= rolling_vol_60d:
        positive.append("Short-term volatility is not above medium-term volatility.")
    else:
        negative.append("Short-term volatility is above medium-term volatility.")

    if beta <= 1.0:
        positive.append("Beta is at or below market sensitivity.")
    elif beta > 1.5:
        negative.append("Beta indicates high market sensitivity.")

    if price_vs_high > -15:
        positive.append("Price remains relatively close to its 52-week high.")
    elif price_vs_high < -30:
        negative.append("Price is far below its 52-week high.")

    if avg_sentiment >= 0.05:
        positive.append("Recent news sentiment is positive.")
    elif avg_sentiment <= -0.05:
        negative.append("Recent news sentiment is negative.")

    return {
        "agent": "Market + Volatility + News Sentiment Agent",
        "ticker": ticker,
        "company": company_name or price_row.get("long_name") or ticker,
        "as_of_date": price_row.get("date"),
        "risk_score": score,
        "risk_label": _label_from_score(score),
        "confidence": 0.80 if price_rows else 0.15,
        "claim_type": "DERIVED_ESTIMATE",
        "metrics": {
            "rolling_vol_30d": round(rolling_vol_30d, 6),
            "rolling_vol_60d": round(rolling_vol_60d, 6),
            "annualized_volatility_pct": round(annual_vol_pct, 2),
            "beta": round(beta, 4),
            "price_vs_52w_high_pct": round(price_vs_high, 2),
            "ytd_return_pct": round(ytd_return, 2),
            "avg_news_sentiment": round(avg_sentiment, 4),
            "recent_article_count": article_count,
        },
        "evidence": [
            f"Annualized 30-day volatility is {annual_vol_pct:.2f}%.",
            f"Beta is {beta:.2f}; price is {price_vs_high:.2f}% versus 52-week high.",
            f"Average recent news sentiment is {avg_sentiment:.3f} across {article_count} articles.",
        ],
        "positive_signals": positive,
        "negative_signals": negative,
        "overall_assessment": (
            f"Market/news risk is {_label_from_score(score)} based on volatility, "
            "beta, price drawdown, YTD return, and news sentiment."
        ),
    }


def run_macro_agent_safe(ticker: str, sector: Optional[str] = None, company_name: Optional[str] = None) -> dict[str, Any]:
    """Use agents.macro_agent if available, otherwise fall back to local rules."""
    ticker = ticker.upper()
    sector = sector or _load_gold_context(ticker).get("sector") or "General"

    if os.getenv("DEEPSEEK_API_KEY"):
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from agents.macro_agent import run_macro_agent

            result = run_macro_agent(ticker=ticker, sector=sector, company_name=company_name or ticker)
            if "macro_risk_score" in result and "risk_score" not in result:
                result["risk_score"] = result["macro_risk_score"]
            result.setdefault("agent", "Macro-Economic Agent")
            result.setdefault("risk_label", _label_from_score(float(result.get("risk_score", 5.0))))
            return result
        except Exception as exc:
            return _local_macro_agent(ticker, sector, company_name, error=str(exc))

    return _local_macro_agent(ticker, sector, company_name)


def _local_macro_agent(
    ticker: str,
    sector: str,
    company_name: Optional[str] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    rows = _read_csv(SILVER_DIR / "silver_macro.csv")
    row = _latest_row(rows, ("date",))

    if not row:
        return {
            "agent": "Macro-Economic Agent",
            "ticker": ticker,
            "company": company_name or ticker,
            "sector": sector,
            "risk_score": 5.0,
            "macro_risk_score": 5.0,
            "risk_label": "MODERATE",
            "confidence": 0.10,
            "evidence": ["Macro data unavailable."],
            "claim_type": "FALLBACK_ESTIMATE",
        }

    fed = _safe_float(row.get("fed_funds_rate"), 0.0) or 0.0
    cpi = _safe_float(row.get("cpi"), 0.0) or 0.0
    treasury = _safe_float(row.get("treasury_10y"), 0.0) or 0.0
    unemployment = _safe_float(row.get("unemployment"), 0.0) or 0.0
    gdp_growth = _safe_float(row.get("gdp_growth"), 0.0) or 0.0

    cpi_prev = None
    if len(rows) > 250:
        cpi_prev = _safe_float(rows[-252].get("cpi"))
    cpi_yoy = ((cpi - cpi_prev) / abs(cpi_prev) * 100.0) if cpi and cpi_prev else 0.0

    score = 3.0
    score += min(max(fed - 2.5, 0.0) / 3.0, 1.0) * 2.0
    score += min(max(cpi_yoy - 2.0, 0.0) / 4.0, 1.0) * 1.5
    score += min(max(unemployment - 4.5, 0.0) / 3.0, 1.0) * 1.5
    score += 1.0 if gdp_growth < 0 else 0.0
    score += 0.5 if treasury > 4.5 else 0.0
    score = _clamp_score(score)

    sensitivity_note = "General macro sensitivity."
    if sector in {"Real Estate", "Utilities"}:
        score = _clamp_score(score + 0.75)
        sensitivity_note = "Rate-sensitive sector; higher rates raise refinancing and valuation pressure."
    elif sector in {"Consumer Staples", "Health Care", "Healthcare"}:
        score = _clamp_score(score - 0.35)
        sensitivity_note = "Defensive sector; macro pressure is partially cushioned by stable demand."
    elif sector in {"Information Technology", "Technology", "Consumer Discretionary"}:
        score = _clamp_score(score + 0.35)
        sensitivity_note = "Growth-sensitive sector; higher rates can pressure valuations."

    evidence = [
        f"Fed funds rate is {fed:.2f}%; 10Y Treasury is {treasury:.2f}%.",
        f"Unemployment is {unemployment:.2f}%; GDP growth is {gdp_growth:.2f}%.",
        f"CPI YoY estimate is {cpi_yoy:.2f}%.",
    ]
    if error:
        evidence.append(f"LLM macro agent unavailable, local macro rules used: {error}")

    return {
        "agent": "Macro-Economic Agent",
        "ticker": ticker,
        "company": company_name or ticker,
        "sector": sector,
        "as_of_date": row.get("date"),
        "risk_score": score,
        "macro_risk_score": score,
        "risk_label": _label_from_score(score),
        "confidence": 0.70,
        "claim_type": "RULE_BASED_ESTIMATE",
        "metrics": {
            "fed_funds_rate": round(fed, 2),
            "cpi": round(cpi, 3),
            "cpi_yoy_pct": round(cpi_yoy, 2),
            "treasury_10y": round(treasury, 2),
            "unemployment": round(unemployment, 2),
            "gdp_growth": round(gdp_growth, 2),
        },
        "evidence": evidence,
        "positive_signals": [],
        "negative_signals": [sensitivity_note] if score > 6 else [],
        "sector_sensitivity": sensitivity_note,
        "overall_assessment": f"Macro risk is {_label_from_score(score)} for {sector}.",
    }


def _weighted_critic(agent_outputs: list[dict[str, Any]], query: str = "") -> dict[str, Any]:
    weights = {
        "Fundamental Agent": 0.40,
        "Market + Volatility + News Sentiment Agent": 0.35,
        "Macro-Economic Agent": 0.25,
    }

    weighted_sum = 0.0
    total_weight = 0.0
    for output in agent_outputs:
        agent = output.get("agent", "")
        score = _safe_float(output.get("risk_score") or output.get("macro_risk_score"))
        if score is None:
            continue
        weight = weights.get(agent, 1.0 / len(agent_outputs))
        weighted_sum += score * weight
        total_weight += weight

    final_score = _clamp_score(weighted_sum / total_weight) if total_weight else 5.0
    final_label = _label_from_score(final_score)
    labels = [o.get("risk_label") for o in agent_outputs if o.get("risk_label")]
    disagreement = len(set(labels)) > 1

    evidence = []
    for output in agent_outputs:
        evidence.append(
            f"{output.get('agent')}: {output.get('risk_label')} "
            f"({output.get('risk_score', output.get('macro_risk_score'))}/10)"
        )

    main_risks = []
    risk_offsets = []
    for output in sorted(
        agent_outputs,
        key=lambda o: _safe_float(o.get("risk_score") or o.get("macro_risk_score"), 0.0) or 0.0,
        reverse=True,
    ):
        for item in output.get("negative_signals", [])[:2]:
            if item not in main_risks:
                main_risks.append(item)
        for item in output.get("positive_signals", [])[:2]:
            if item not in risk_offsets:
                risk_offsets.append(item)

    return {
        "agent": "LLM-Based Critic Agent",
        "critic_type": "llm_based_critic_with_local_fallback",
        "query": query,
        "final_risk_score": final_score,
        "final_risk_label": final_label,
        "confidence": 0.72 if not disagreement else 0.60,
        "disagreement_detected": disagreement,
        "agent_scorecard": evidence,
        "main_risk_drivers": main_risks[:5],
        "risk_offsets": risk_offsets[:5],
        "final_decision": (
            f"Final risk is {final_label} at {final_score}/10. "
            "The critic combined fundamental strength, market/news pressure, and macro conditions."
        ),
    }


def _llm_critic(agent_outputs: list[dict[str, Any]], query: str) -> Optional[dict[str, Any]]:
    """Optional OpenAI-compatible critic. Returns None when not configured."""
    api_key = os.getenv("CRITIC_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("CRITIC_BASE_URL")
    model = os.getenv("CRITIC_MODEL", "gpt-4o-mini")
    if os.getenv("DEEPSEEK_API_KEY") and not os.getenv("CRITIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        base_url = base_url or "https://api.deepseek.com"
        model = os.getenv("CRITIC_MODEL", "deepseek-chat")

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        prompt = {
            "query": query,
            "agent_outputs": agent_outputs,
            "required_output": {
                "agent": "LLM-Based Critic Agent",
                "final_risk_score": "number 1-10",
                "final_risk_label": "LOW/MODERATE/HIGH",
                "confidence": "0-1",
                "disagreement_detected": "boolean",
                "agent_scorecard": ["short summary per agent"],
                "main_risk_drivers": ["top reasons"],
                "risk_offsets": ["positive reasons"],
                "final_decision": "clear final answer",
            },
        }
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial critic agent. Compare the three agent outputs, "
                        "resolve disagreements, and return strict JSON only."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, indent=2)},
            ],
        )
        text = response.choices[0].message.content or ""
        start, end = text.find("{"), text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(text[start:end])
            result["critic_type"] = f"llm:{model}"
            return result
    except Exception as exc:
        return {
            "agent": "LLM-Based Critic Agent",
            "critic_type": "llm_failed_fallback_needed",
            "error": str(exc),
        }

    return None


def run_critic_agent(
    ticker: str,
    query: str = "",
    company_name: Optional[str] = None,
    sector: Optional[str] = None,
    use_llm: bool = False,
) -> dict[str, Any]:
    ticker = ticker.upper()
    gold = _load_gold_context(ticker)
    sector = sector or gold.get("sector") or "General"
    company_name = company_name or ticker

    outputs = [
        run_fundamental_agent(ticker, company_name=company_name),
        run_market_sentiment_agent(ticker, company_name=company_name),
        run_macro_agent_safe(ticker, sector=sector, company_name=company_name),
    ]

    llm_result = _llm_critic(outputs, query) if use_llm else None
    if llm_result and llm_result.get("critic_type") != "llm_failed_fallback_needed":
        final = llm_result
    else:
        final = _weighted_critic(outputs, query=query)
        if llm_result and llm_result.get("error"):
            final["llm_error"] = llm_result["error"]

    return {
        "ticker": ticker,
        "query": query,
        "agent_outputs": outputs,
        "final_output": final,
    }


def print_critic_report(report: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print(f"CRITIC AGENT REPORT - {report['ticker']}")
    print("=" * 72)

    for output in report["agent_outputs"]:
        print(f"\n[{output.get('agent')}]")
        print(json.dumps(output, indent=2))

    print("\n[FINAL LLM-BASED CRITIC OUTPUT]")
    print(json.dumps(report["final_output"], indent=2))
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all three agents and a critic agent for one ticker.")
    parser.add_argument("ticker", help="Ticker symbol, for example AAPL or A")
    parser.add_argument("--query", default="", help="Optional natural-language query for the critic.")
    parser.add_argument("--company", default=None, help="Optional company name.")
    parser.add_argument("--sector", default=None, help="Optional sector.")
    parser.add_argument("--use-llm", action="store_true", help="Use configured LLM critic instead of weighted fallback.")
    parser.add_argument("--save", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    report = run_critic_agent(
        ticker=args.ticker,
        query=args.query,
        company_name=args.company,
        sector=args.sector,
        use_llm=args.use_llm,
    )
    print_critic_report(report)

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved report to {save_path}")


if __name__ == "__main__":
    main()
