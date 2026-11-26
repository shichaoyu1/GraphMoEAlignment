import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest


@pytest.mark.parametrize("experiment", ["idea3_role_transition", "idea3_record_value"])
def test_background_launcher_uses_uploaded_archives_and_separate_run_directory(tmp_path, experiment):
    bash = shutil.which("bash")
    if not bash and os.name == "nt" and shutil.which("git"):
        bash = Path(shutil.which("git")).parents[1] / "bin" / "bash.exe"
    if not bash or not Path(bash).exists():
        pytest.skip("Bash unavailable")
    repo = tmp_path / "glioma"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    launcher = scripts / f"run_{experiment}_autodl.sh"
    shutil.copyfile(Path(__file__).resolve().parents[1] / "scripts" / launcher.name, launcher)
    archive = repo / "output" / "idea3_rule_mri_swap"
    for name in ("utsw_authority_grade/manifest.json", "utsw_authority_grade/image_audit.json",
                 "experiment2/authority_oof_predictions.csv", "experiment2/fold0/best.pt"):
        path = archive / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (scripts / "run_idea3_module.py").write_text(
        "import json, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "out = pathlib.Path(args[args.index('--output') + 1])\n"
        "print('background test')\n"
        "(out / 'args.json').write_text(json.dumps(args))\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("IDEA3_")}
    env["IDEA3_PYTHON"] = Path(sys.executable).as_posix()
    for _ in range(2):
        proc = subprocess.run([str(bash), launcher.as_posix()], cwd=repo, capture_output=True, text=True, env=env)
        assert proc.returncode == 0, proc.stderr
        assert "PID " in proc.stdout
    runs = list((repo / "output" / experiment).glob("run_*"))
    assert len(runs) == 2
    for run in runs:
        for _ in range(150):
            if (run / "args.json").is_file() and (run / "args.json").stat().st_size:
                break
            time.sleep(.1)
        assert (run / "run.pid").read_text().strip().isdigit()
        assert "background test" in (run / "run.log").read_text()
        args = json.loads((run / "args.json").read_text())
        assert args[0] == f"glioma.cli.{experiment}"
        for option, path in (("--authority-source", archive / "utsw_authority_grade"),
                             ("--residual-source", archive / "experiment2"), ("--output", run)):
            assert Path(args[args.index(option) + 1]).resolve() == path.resolve()
