import os
import subprocess
import sys


def test_cli_import_does_not_import_sklearn_linear_model():
    script = """
import importlib.abc
import sys

class BlockSklearnLinearModel(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sklearn.linear_model":
            raise RuntimeError("sklearn.linear_model import blocked")
        return None

sys.meta_path.insert(0, BlockSklearnLinearModel())
import quantforge.cli
print("cli ok")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(
        [sys.executable, "-u", "-c", script],
        cwd=os.getcwd(),
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "cli ok"
