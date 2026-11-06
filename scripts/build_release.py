"""Build a credential-free Windows portable release using the current interpreter."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def collect_notices(destination: Path) -> None:
    packages = []
    for distribution in sorted(
        importlib.metadata.distributions(), key=lambda d: d.metadata["Name"].lower()
    ):
        name = distribution.metadata["Name"]
        packages.append({"name": name, "version": distribution.version})
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        for entry in distribution.files or ():
            parts = Path(str(entry)).parts
            if not any(part.endswith(".dist-info") for part in parts):
                continue
            if not any(
                part.lower().startswith(("license", "copying", "notice"))
                for part in parts
            ):
                continue
            source = Path(distribution.locate_file(entry))
            if not source.is_file():
                continue
            info_index = next(i for i, part in enumerate(parts) if part.endswith(".dist-info"))
            relative = Path(*parts[info_index + 1 :])
            target = destination / "THIRD_PARTY_LICENSES" / safe_name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    write_atomic(destination / "BUILD_DEPENDENCIES.json", json.dumps(packages, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Release version, such as 0.1.0")
    parser.add_argument("--include-base-model", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path(r"G:\Program Files\codexcache"))
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("Build the Windows release on Windows.")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9.-]+)?", args.version):
        parser.error("Invalid version format.")

    output = ROOT / "release" / f"v{args.version}"
    if output.exists():
        parser.error(f"Output already exists; move it aside before rebuilding: {output}")
    model = ROOT / "models" / "whisper-base"
    if args.include_base_model:
        for name in ("config.json", "model.bin", "tokenizer.json"):
            if not (model / name).is_file():
                parser.error(f"Missing base model file: {model / name}")
        if not (ROOT / "docs" / "licenses" / "whisper-LICENSE.txt").is_file():
            parser.error("Missing upstream Whisper license.")

    cache = args.cache_dir.resolve()
    temporary = cache / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        TEMP=str(temporary), TMP=str(temporary),
        PYINSTALLER_CONFIG_DIR=str(cache / "pyinstaller"),
        PIP_CACHE_DIR=str(cache / "pip"),
    )
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
         "--distpath", str(output),
         "--workpath", str(cache / "trans-publish" / f"build-{args.version}"),
         str(ROOT / "TransSubtitle.spec")],
        cwd=ROOT, env=environment, check=True,
    )
    application = output / "TransSubtitle"
    # Public templates are the only allowed configuration source.
    settings = json.loads((ROOT / "settings.example.json").read_text(encoding="utf-8"))
    if settings.get("deepseek_api_key"):
        raise ValueError("The public settings template must not contain an API key.")
    settings.update(translator_backend="api", whisper_model="base", cached_models=[])
    write_atomic(application / "settings.json", json.dumps(settings, indent=2) + "\n")
    for name in ("README.md", "THIRD_PARTY_NOTICES.md"):
        shutil.copy2(ROOT / name, application / name)
    if args.include_base_model:
        shutil.copytree(model, application / "models" / "whisper-base")
        shutil.copy2(ROOT / "docs" / "licenses" / "whisper-LICENSE.txt",
                     application / "models" / "whisper-base" / "LICENSE.txt")
    collect_notices(application)
    for path in application.rglob("*"):
        if path.is_file() and (
            path.name in {".env", ".env.local"} or path.suffix.lower() in {".pfx", ".p12", ".key"}
        ):
            raise ValueError(f"Private file in release: {path.relative_to(application)}")
    archive = output / f"TransSubtitle-{args.version}-windows-x64.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(application.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(output))
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    write_atomic(output / "SHA256SUMS.txt", f"{digest}  {archive.name}\n")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "sha256": digest}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
