"""Cached transfers and safe archive extraction for public dataset sources."""
from contextlib import contextmanager
from http.client import HTTPException
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tarfile
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

from tqdm import tqdm


@contextmanager
def preparation_lock(root):
    """Prevent two setup processes from writing the same cache/manifests."""
    import fcntl
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another preparation is running in {root}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def download(url, destination, *, checksum=None, attempts=3):
    """Resume an interrupted HTTP transfer; expose only complete files."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.stat().st_size == 0:
            raise ValueError(f"Empty cached download: {destination}")
        if checksum and _digest(destination) != checksum:
            raise ValueError(f"Checksum mismatch: {destination}")
        return destination
    partial = destination.with_name(destination.name + ".part")
    metadata = partial.with_name(partial.name + ".json")
    for attempt in range(attempts):
        try:
            saved = json.loads(metadata.read_text()) if metadata.exists() else {}
            offset = partial.stat().st_size if partial.exists() and saved.get("url") == url and saved.get("validator") else 0
            headers = {"User-Agent": "MPSL-dataset-setup/1.0", "Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
                if saved.get("validator"):
                    headers["If-Range"] = saved["validator"]
            with urlopen(Request(url, headers=headers), timeout=30) as response:
                content_type = response.headers.get("Content-Type", "").lower()
                if "text/html" in content_type or "application/xhtml" in content_type:
                    raise ValueError(f"Download returned a web page, not a data file: {url}")
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if not match or int(match[1]) != offset or not offset <= int(match[2]) < int(match[3]):
                        raise ValueError(f"Unexpected download range: {content_range}")
                    total = int(match[3])
                else:
                    offset = 0  # A server may ignore Range or the file may have changed.
                    length = response.headers.get("Content-Length")
                    total = int(length) if length is not None else None
                if total and total - offset > shutil.disk_usage(destination.parent).free:
                    raise OSError(f"Not enough disk space for {destination.name} ({total:,} bytes)")
                metadata.write_text(json.dumps({"url": url, "validator": response.headers.get("ETag") or response.headers.get("Last-Modified")}))
                with partial.open("ab" if offset else "wb") as output, tqdm(total=total, initial=offset, unit="B", unit_scale=True, desc=destination.name, disable=None) as progress:
                    while block := response.read(1024 * 1024):
                        output.write(block)
                        progress.update(len(block))
                if total is not None and partial.stat().st_size != total:
                    raise OSError(f"Incomplete download: {destination.name}")
            if not partial.stat().st_size:
                raise ValueError(f"Empty download: {url}")
            if checksum and _digest(partial) != checksum:
                partial.unlink()
                raise ValueError(f"Checksum mismatch: {url}")
            partial.replace(destination)
            metadata.unlink(missing_ok=True)
            return destination
        except HTTPError as error:
            if error.code == 416 and offset:
                # Re-fetch rather than assume an unverified partial file is complete.
                partial.unlink(missing_ok=True)
                metadata.unlink(missing_ok=True)
            elif error.code < 500 or attempt == attempts - 1:
                raise
        except (URLError, TimeoutError, ConnectionError, OSError, HTTPException):
            if attempt == attempts - 1:
                raise OSError(f"Download interrupted: {url}. Rerun to resume.")
        time.sleep(attempt + 1)
    raise OSError(f"Download failed: {url}")


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _target(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or (relative.parts and ":" in relative.parts[0]):
        raise ValueError(f"Unsafe archive path: {name}")
    target = root / relative
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Archive path escapes destination: {name}")
    if target.is_symlink():
        raise ValueError(f"Archive target is a symlink: {target}")
    return target


def _write_member(target, blocks, expected_size):
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".mpsl-extract-", dir=target.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            for block in blocks:
                stream.write(block)
        if temporary.stat().st_size != expected_size:
            raise ValueError(f"Truncated archive member: {target}")
        if target.exists():
            if not target.is_file() or target.stat().st_size != expected_size or _digest(target) != _digest(temporary):
                raise FileExistsError(f"Existing file differs from archive; keep or relocate it before retrying: {target}")
        else:
            temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def extract_archive(archive, destination, *, selected=None):
    """Extract regular files only; optionally map exact basenames to destinations."""
    errors = (tarfile.TarError, zipfile.BadZipFile, EOFError)
    if Path(archive).suffix.lower() == ".rar":
        try:
            from libarchive.exception import ArchiveError
        except (ImportError, OSError) as error:
            raise RuntimeError("RAR extraction needs libarchive: brew install libarchive (macOS), or install your Linux distribution's libarchive package.") from error
        errors += (ArchiveError,)
    try:
        return _extract_archive(archive, destination, selected=selected)
    except errors as error:
        raise ValueError(f"Unreadable archive: {archive}. Remove or replace it and rerun.") from error


def _extract_archive(archive, destination, *, selected):
    archive, destination = Path(archive), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    written = []

    def target_for(name):
        _target(destination, name)  # Validate even unselected names.
        if selected is not None:
            name = selected.get(Path(name).name)
            if name is None:
                return None
        return _target(destination, name)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as stream:
            for entry in stream.infolist():
                target = target_for(entry.filename)
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError(f"Archive links are not supported: {entry.filename}")
                if entry.is_dir() or target is None:
                    continue
                with stream.open(entry) as data:
                    _write_member(target, iter(lambda: data.read(1024 * 1024), b""), entry.file_size)
                written.append(target)
    elif archive.suffix.lower() == ".rar":
        import libarchive
        with libarchive.file_reader(str(archive)) as stream:
            for entry in stream:
                target = target_for(entry.pathname)
                if entry.isdir:
                    continue
                if not entry.isfile or entry.linkpath:
                    raise ValueError(f"Archive links/special files are not supported: {entry.pathname}")
                if target is not None:
                    _write_member(target, entry.get_blocks(), entry.size)
                    written.append(target)
    else:
        with tarfile.open(archive, mode="r|*") as stream:
            for entry in stream:
                target = target_for(entry.name)
                if entry.isdir():
                    continue
                if not entry.isfile():
                    raise ValueError(f"Archive links/special files are not supported: {entry.name}")
                if target is not None:
                    with stream.extractfile(entry) as data:
                        _write_member(target, iter(lambda: data.read(1024 * 1024), b""), entry.size)
                    written.append(target)
    return written
