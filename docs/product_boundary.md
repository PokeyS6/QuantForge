# Product Boundary

QuantForge is a local-first, CLI-first quantitative strategy evaluation
workbench with optional ML-based filters. Users own the hypothesis. QuantForge
helps test, modify, validate, compare, and audit user-provided strategy ideas.

QuantForge does not provide trading advice. It does not make buy/sell
recommendations, place live trades, connect to brokers for execution, promise
profitable outcomes, or provide performance guarantees.

## Operating Principles

- Users provide the strategy hypothesis and decide what to test.
- QuantForge helps compare strategies; it does not tell users what to buy or sell.
- No trading advice is provided.
- No performance guarantees are provided.
- No live trading or broker execution is part of the product.
- No hidden optimization is allowed.
- Generated code, assumptions, parameters, and diffs must be visible.
- Every modification is treated as an experiment, not a recommendation.
- Results should be framed as evidence from a test, not as instructions to trade.
- Filters can reduce participation or change exposure; they do not guarantee
  lower drawdown.

## Safe Language Examples

- "This variant reduced drawdown in the tested period."
- "The volatility filter changed exposure during high-volatility regimes."
- "Here is the diff between the baseline and the variant."
- "This result depends on the selected data, assumptions, and validation window."
- "No strategy recommendation is being made."

## Unsafe Language Examples

- "Buy AAPL now."
- "This is the best strategy."
- "Use these parameters to maximize profit."
- "This signal predicts tomorrow's move."
- "QuantForge recommends this trade."
- "This variant is guaranteed to outperform."
