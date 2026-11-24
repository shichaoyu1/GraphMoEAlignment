"""Run a Glioma module from a checkout with any directory name."""

import importlib.util
from pathlib import Path
import runpy
import sys


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_idea3_module.py MODULE [ARGS...]")
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("glioma", root / "__init__.py",
                                                  submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules["glioma"] = package
    spec.loader.exec_module(package)
    module = sys.argv.pop(1)
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
