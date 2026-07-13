#!/usr/bin/env python3
"""Build a deterministic portable Hanabi Workshop archive."""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import stat
import tarfile
import tempfile
import zipfile
from pathlib import Path

REPOSITORY_FILES = (
    "README.md",
    "CHANGELOG.md",
    "LICENSE-APACHE2",
    "LICENSE-MIT",
    "THIRD_PARTY_LICENSES.txt",
)
ZIP_MIN_EPOCH_SECONDS = 315_532_800


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--platform", choices=("linux", "windows", "macos"), required=True
    )
    parser.add_argument("--target", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dist"))
    return parser.parse_args()


def copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(".DS_Store"),
    )


def stage_portable(root: Path, binary: Path, platform: str) -> None:
    executable = root / (
        "hanabi-workshop.exe" if platform == "windows" else "hanabi-workshop"
    )
    shutil.copy2(binary, executable)
    executable.chmod(
        executable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )
    copy_tree(Path("assets"), root / "assets")
    copy_tree(Path("examples"), root / "examples")
    for source in REPOSITORY_FILES:
        shutil.copy2(source, root / source)
    if platform == "linux":
        shutil.copy2("hanabi.magic", root / "hanabi.magic")


def stage_macos(root: Path, binary: Path, version: str) -> None:
    contents = root / "Hanabi Workshop.app" / "Contents"
    executable = contents / "MacOS" / "hanabi-workshop"
    resources = contents / "Resources"
    executable.parent.mkdir(parents=True)
    resources.mkdir(parents=True)
    shutil.copy2(binary, executable)
    executable.chmod(
        executable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )
    copy_tree(Path("assets"), resources / "assets")
    copy_tree(Path("examples"), resources / "examples")
    for source in REPOSITORY_FILES:
        shutil.copy2(source, resources / source)

    (contents / "Info.plist").write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "https://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDisplayName</key>
    <string>Hanabi Workshop</string>
    <key>CFBundleExecutable</key>
    <string>hanabi-workshop</string>
    <key>CFBundleIdentifier</key>
    <string>fr.djee.hanabi-workshop</string>
    <key>CFBundleName</key>
    <string>Hanabi Workshop</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>{version}</string>
    <key>CFBundleVersion</key>
    <string>{version}</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
""",
        encoding="utf-8",
    )


def archive_paths(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.name != ".DS_Store"),
        key=lambda path: path.as_posix(),
    )


def write_zip(root: Path, destination: Path, epoch: int) -> None:
    timestamp = tuple(
        __import__("time").gmtime(max(epoch, ZIP_MIN_EPOCH_SECONDS))[:6]
    )
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in archive_paths(root):
            relative = path.relative_to(root.parent).as_posix()
            info = zipfile.ZipInfo(relative + ("/" if path.is_dir() else ""), timestamp)
            info.create_system = 3
            mode = path.stat().st_mode
            info.external_attr = (mode & 0xFFFF) << 16
            if path.is_dir():
                archive.writestr(info, b"")
            else:
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes())


def write_tar_gz(root: Path, destination: Path, epoch: int) -> None:
    with destination.open("wb") as output:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=output, mtime=epoch, compresslevel=9
        ) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in [root, *archive_paths(root)]:
                    relative = path.relative_to(root.parent)
                    info = archive.gettarinfo(path, arcname=relative)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = epoch
                    if path.is_file():
                        with path.open("rb") as source:
                            archive.addfile(info, source)
                    else:
                        archive.addfile(info)


def verify_archive(destination: Path, platform: str, archive_root: str) -> None:
    if destination.suffix == ".zip":
        with zipfile.ZipFile(destination) as archive:
            names = archive.namelist()
            modes = {
                info.filename.rstrip("/"): info.external_attr >> 16
                for info in archive.infolist()
            }
            with tempfile.TemporaryDirectory() as temporary:
                archive.extractall(temporary)
                extracted_root = Path(temporary)
                verify_extracted(extracted_root, platform, archive_root)
    else:
        with tarfile.open(destination, "r:gz") as archive:
            names = archive.getnames()
            modes = {member.name: member.mode for member in archive.getmembers()}
            with tempfile.TemporaryDirectory() as temporary:
                archive.extractall(temporary)
                extracted_root = Path(temporary)
                verify_extracted(extracted_root, platform, archive_root)

    if any(Path(name).name == ".DS_Store" for name in names):
        raise RuntimeError("archive contains .DS_Store")

    prefix = (
        "Hanabi Workshop.app/Contents/Resources"
        if platform == "macos"
        else archive_root
    )
    required = (
        f"{prefix}/assets/",
        f"{prefix}/examples/",
        f"{prefix}/THIRD_PARTY_LICENSES.txt",
    )
    for expected in required:
        if not any(
            name == expected.rstrip("/") or name.startswith(expected) for name in names
        ):
            raise RuntimeError(f"archive is missing {expected}")

    if platform == "macos":
        executable = "Hanabi Workshop.app/Contents/MacOS/hanabi-workshop"
    elif platform == "windows":
        executable = f"{archive_root}/hanabi-workshop.exe"
    else:
        executable = f"{archive_root}/hanabi-workshop"
    if executable not in names:
        raise RuntimeError(f"archive is missing {executable}")
    if platform != "windows" and modes[executable] & 0o111 == 0:
        raise RuntimeError(f"{executable} is not executable")


def verify_extracted(root: Path, platform: str, archive_root: str) -> None:
    prefix = (
        root / "Hanabi Workshop.app" / "Contents" / "Resources"
        if platform == "macos"
        else root / archive_root
    )
    for expected in (
        prefix / "assets",
        prefix / "examples",
        prefix / "THIRD_PARTY_LICENSES.txt",
    ):
        if not expected.exists():
            raise RuntimeError(f"extracted archive is missing {expected.relative_to(root)}")

    if platform == "linux" and not (prefix / "hanabi.magic").is_file():
        raise RuntimeError("extracted Linux archive is missing hanabi.magic")

    if platform == "macos" and not (
        root / "Hanabi Workshop.app" / "Contents" / "Info.plist"
    ).is_file():
        raise RuntimeError("extracted macOS archive is missing Info.plist")


def main() -> int:
    args = parse_args()
    if not args.binary.is_file():
        raise FileNotFoundError(args.binary)

    args.output.mkdir(parents=True, exist_ok=True)
    archive_root = f"hanabi-workshop-v{args.version}-{args.target}"
    suffix = ".tar.gz" if args.platform == "linux" else ".zip"
    destination = args.output / f"{archive_root}{suffix}"
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))

    with tempfile.TemporaryDirectory(dir=args.output) as temporary:
        temporary_root = Path(temporary)
        if args.platform == "macos":
            stage_root = temporary_root / "bundle"
            stage_macos(stage_root, args.binary, args.version)
            archive_content = stage_root / "Hanabi Workshop.app"
            write_zip(archive_content, destination, epoch)
        else:
            archive_content = temporary_root / archive_root
            archive_content.mkdir()
            stage_portable(archive_content, args.binary, args.platform)
            if args.platform == "linux":
                write_tar_gz(archive_content, destination, epoch)
            else:
                write_zip(archive_content, destination, epoch)

    verify_archive(destination, args.platform, archive_root)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
