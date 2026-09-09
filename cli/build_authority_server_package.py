"""Create a source-only, checksummed server archive; never package patient data."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from glioma.cli.run_authority_protocol import development_jobs, formal_jobs
from glioma.cli.train_authority import source_hash


def build(output):
    root = Path(__file__).resolve().parents[1]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    files = {}
    # Explicit source directories, not an unrestricted recursive workspace archive.
    folders = ("anchors", "cli", "config", "data", "eval", "io", "models", "modules", "objectives",
               "semantic", "tests", "training", "validation", "visualization")
    for folder in folders:
        for path in sorted((root / folder).rglob("*.py")):
            if "__pycache__" not in path.parts:
                files["glioma/" + path.relative_to(root).as_posix()] = path.read_bytes()
    for name in ("__init__.py", "authority_server.sh", "autodl_authority.sh", "AUTODL_AUTHORITY.md",
                 "run_server_paper4_geodesic_full.sh", "requirements-authority.txt",
                 "requirements-authority-server.txt", "PAPER4_AUTHORITY_GUIDE.md"):
        files["glioma/" + name] = (root / name).read_bytes()
    files["README.md"] = ("# ACF-SPD server package\n\n"
                           "Start with glioma/PAPER4_AUTHORITY_GUIDE.md. Keep the folder name glioma.\n"
                           "510 formal trainings are prepared, not executed. Run development and freeze first.\n"
                           "Contains source and planned manifests only; no MRI data or existing experimental outputs.\n").encode()
    for name, jobs in (("development_manifest", development_jobs()), ("main_planned", formal_jobs("main")),
                       ("attribution_planned", formal_jobs("attribution"))):
        files[f"glioma/reports/paper4_authority_v1/{name}.json"] = json.dumps(jobs, indent=2).encode()
    manifest = dict(source_hash=source_hash(), main_trainings=300, attribution_trainings=210, development_trials=60,
                    files={name: hashlib.sha256(content).hexdigest() for name, content in sorted(files.items())})
    files["PACKAGE_MANIFEST.json"] = json.dumps(manifest, indent=2).encode()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026,9,9,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if name.endswith(".sh") else 0o644) << 16
            archive.writestr(info, content)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive CRC validation failed")
        for name, expected in manifest["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise RuntimeError("Archive content hash mismatch: " + name)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(digest + "  " + output.name + "\n", encoding="utf-8")
    print(json.dumps(dict(path=str(output.resolve()), bytes=output.stat().st_size, files=len(files), sha256=digest,
                          source_hash=manifest["source_hash"]), indent=2))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] /
                        "reports/paper4_authority_v1/acf_spd_authority_server_20260909.zip")
    args = parser.parse_args(argv)
    build(args.output)


if __name__ == "__main__":
    main()
