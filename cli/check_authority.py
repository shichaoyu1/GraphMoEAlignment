"""Run authority acceptance and legacy regression checks before server training."""

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from glioma.cli.train_authority import atomic_json, source_hash


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-missing-general", action="store_true",
                        help="Explicitly allow analytic-only checks with the optional solver test skipped")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    output = args.output or root / "reports/paper4_authority_v1/preflight"
    output.mkdir(parents=True, exist_ok=True)
    installed = importlib.util.find_spec("cvxpylayers") is not None
    if not installed and not args.allow_missing_general:
        parser.error("Install requirements-authority-server.txt in Python 3.11+ to include general solver validation")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root.parent) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    command = [sys.executable, "-m", "pytest", "tests/test_authority.py", "tests/test_authority_protocol.py",
               "tests/test_paper4_iclr.py", "tests/test_geodesic_fusion.py", "-q", f"--junitxml={output.resolve() / 'tests.xml'}"]
    result = subprocess.run(command, cwd=root, env=env)
    atomic_json(output / "status.json", dict(source_hash=source_hash(), returncode=result.returncode,
                                             general_solver_installed=installed, python=sys.version,
                                             scope="Synthetic authority acceptance and existing benchmark/SPD regressions"))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
