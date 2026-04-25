"""Core data models for QuantForge."""

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class StrategyVariant:
    """Metadata for a strategy variant."""

    variant_id: str
    parent_variant_id: str | None
    user_instruction: str
    strategy_type: str
    parameters: dict[str, Any]
    assumptions: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrategyProject:
    """Metadata for a local QuantForge strategy project."""

    project_schema_version: str
    strategy_id: str
    name: str
    created_at: str
    asset_universe: list[str]
    timeframe: str
    data_window: dict[str, str | None]
    baseline_variant_id: str
    variants: list[StrategyVariant]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_rsi_project(strategy_prompt: str, ticker: str, start: str) -> StrategyProject:
    """Create metadata for the first constrained RSI project slice."""
    normalized_ticker = ticker.upper()
    baseline_variant_id = "baseline"
    variant = StrategyVariant(
        variant_id=baseline_variant_id,
        parent_variant_id=None,
        user_instruction=strategy_prompt,
        strategy_type="rsi_reversal",
        parameters={
            "ticker": normalized_ticker,
            "start": start,
            "entry_rsi": 30,
            "exit_rsi": 70,
        },
        assumptions=[
            "User owns the strategy hypothesis.",
            "QuantForge is persisting a test configuration, not recommending a trade.",
            "No buy/sell recommendation is being made.",
        ],
        warnings=[
            "QuantForge is not a financial advisor.",
            "This project is not connected to live trading or broker execution.",
        ],
    )
    return StrategyProject(
        project_schema_version="0.1",
        strategy_id=f"rsi_reversal_{normalized_ticker.lower()}",
        name=f"{normalized_ticker} RSI reversal",
        created_at=datetime.now(UTC).isoformat(),
        asset_universe=[normalized_ticker],
        timeframe="1d",
        data_window={"start": start, "end": None},
        baseline_variant_id=baseline_variant_id,
        variants=[variant],
    )
