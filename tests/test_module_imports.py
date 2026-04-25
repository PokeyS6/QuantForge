import importlib


MODULES = [
    "quantforge.core",
    "quantforge.core.models",
    "quantforge.core.project_io",
    "quantforge.core.paths",
    "quantforge.core.warnings",
    "quantforge.data",
    "quantforge.data.loader",
    "quantforge.data.csv_loader",
    "quantforge.data.yfinance_loader",
    "quantforge.strategies",
    "quantforge.strategies.indicators",
    "quantforge.strategies.rsi",
    "quantforge.strategies.templates",
    "quantforge.backtest",
    "quantforge.backtest.engine",
    "quantforge.backtest.metrics",
    "quantforge.backtest.trades",
    "quantforge.variants",
    "quantforge.variants.mutations",
    "quantforge.variants.diffs",
    "quantforge.variants.registry",
    "quantforge.validation",
    "quantforge.validation.checks",
    "quantforge.validation.leakage",
    "quantforge.validation.assumptions",
    "quantforge.reports",
    "quantforge.reports.markdown",
    "quantforge.reports.json_report",
    "quantforge.reports.comparison",
    "quantforge.ml",
    "quantforge.ml.price_floor",
    "quantforge.ml.walk_forward",
]


def test_placeholder_modules_import():
    for module_name in MODULES:
        importlib.import_module(module_name)
