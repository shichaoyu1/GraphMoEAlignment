"""Bundle source for upload, or audited outputs for return; never bundle raw MRI."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def package_source(destination):
    root = Path(__file__).resolve().parents[1]
    files = [root/"__init__.py", root/"requirements.txt",
             root/"docs"/"paper4"/"utsw-authority.md", root/"tests"/"test_utsw_authority.py"]
    for folder in ("data", "models", "modules", "eval", "cli", "visualization", "training",
                   "objectives", "anchors", "io", "config", "semantic", "paper2_revision"):
        if (root/folder).exists():
            files.extend((root/folder).rglob("*.py"))
    files.extend((root/"config").glob("utsw_authority*.tex"))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with zipfile.ZipFile(destination,"w",zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            relative = "glioma/"+path.relative_to(root).as_posix()
            content = path.read_bytes()
            archive.writestr(relative,content)
            hashes[relative]=hashlib.sha256(content).hexdigest()
        archive.writestr("SOURCE_SHA256.json",json.dumps(hashes,indent=2))
    print(f"{destination.resolve()} ({len(hashes)} files)")


def package_results(output):
    out=Path(output)
    if not (out/"oof_predictions.csv").exists() or not (out/"figure_provenance.json").exists():
        raise ValueError("Aggregate complete formal results before packaging")
    destination=out/"results_for_paper.zip"
    selected=json.loads((out/"figure_provenance.json").read_text())["subject_id"]
    files=[p for p in out.rglob("*") if p.is_file() and p.suffix in (".csv",".json",".tex",".md",".pdf",".png")
           and "cache" not in p.relative_to(out).parts]
    files.extend(out.glob("fold*/*/states.npz"))
    # A single cached ROI is sufficient to regenerate the paired figure.
    files.append(out/"cache"/f"{selected}.npz")
    with zipfile.ZipFile(destination,"w",zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path,path.relative_to(out))
    print(destination.resolve())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind",choices=["source","results"])
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    (package_source if args.kind=="source" else package_results)(args.output)
