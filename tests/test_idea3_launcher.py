from pathlib import Path
import shutil
import subprocess
import sys


def test_module_bootstrap_from_renamed_checkout(tmp_path):
    checkout = tmp_path / "GraphMoEAlignment"
    scripts = checkout / "scripts"
    scripts.mkdir(parents=True)
    launcher = Path(__file__).resolve().parents[1] / "scripts" / "run_idea3_module.py"
    shutil.copyfile(launcher, scripts / launcher.name)
    (checkout / "__init__.py").write_text("VALUE = 42\n", encoding="utf-8")
    (checkout / "check.py").write_text(
        "from glioma import VALUE\nimport sys\nprint(VALUE, sys.argv[1])\n", encoding="utf-8"
    )
    result = subprocess.run([sys.executable, "-I", str(scripts / launcher.name), "glioma.check", "argument"],
                            cwd=tmp_path, check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "42 argument"
    assert not (checkout / "output").exists()
