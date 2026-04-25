from typer.testing import CliRunner

from quantforge.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "create" in result.output
    assert "analyze" in result.output
    assert "modify" in result.output
    assert "compare" in result.output


def test_create_help():
    result = runner.invoke(app, ["create", "--help"])

    assert result.exit_code == 0


def test_analyze_help():
    result = runner.invoke(app, ["analyze", "--help"])

    assert result.exit_code == 0


def test_modify_help():
    result = runner.invoke(app, ["modify", "--help"])

    assert result.exit_code == 0


def test_compare_help():
    result = runner.invoke(app, ["compare", "--help"])

    assert result.exit_code == 0
