"""One-command setup, safe transfers, and repeatable source preparation."""
import io
import json
from pathlib import Path
import shutil
import tarfile
import zipfile

import pytest

from test_prepared_datasets import source_fixture
from mpsl.available_datasets import preparation


def test_dataset_setup_uses_defaults_and_can_be_repeated(tmp_path, monkeypatch):
    source, _ = source_fixture(tmp_path, "coco-qa")
    source.rename(tmp_path / "raw")
    monkeypatch.chdir(tmp_path)
    for _ in range(2):
        assert preparation.main(["--dataset", "coco-qa", "--no-download", "--skip-weights"]) == 0
    manifest = json.loads((tmp_path / "data/coco-qa/manifest.json").read_text())
    assert manifest["counts"] == {"train": 1, "test": 1}


def test_all_mode_continues_after_missing_manual_data(tmp_path, monkeypatch, capsys):
    source, _ = source_fixture(tmp_path, "coco-qa")
    source.rename(tmp_path / "raw")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(preparation, "DATASETS", ("t4sa", "coco-qa"))
    assert preparation.main(["--no-download", "--skip-weights"]) == 1
    assert (tmp_path / "data/coco-qa/manifest.json").is_file()
    assert not (tmp_path / "data/t4sa/manifest.json").exists()
    output = capsys.readouterr().out
    assert "t4sa" in output and "coco-qa" in output


def test_validate_never_attempts_acquisition(tmp_path, monkeypatch):
    source, _ = source_fixture(tmp_path, "coco-qa")
    preparation.prepare_dataset("coco-qa", tmp_path / "data", source)
    monkeypatch.chdir(tmp_path)
    assert preparation.main(["--dataset", "coco-qa", "--validate"]) == 0
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.parametrize("partially_extracted", [False, True])
def test_automatic_coco_sources_feed_real_manifest(tmp_path, monkeypatch, partially_extracted):
    from mpsl.available_datasets import acquisition
    source, _ = source_fixture(tmp_path, "coco-qa")
    cache = tmp_path / "raw/.downloads"
    cache.mkdir(parents=True)
    # Cache complete source archives, so this exercises the offline extraction path.
    for key, folder in (("cocoqa", "coco-qa"), ("coco-train", "ms-coco/train2014"), ("coco-val", "ms-coco/val2014")):
        target = cache / acquisition.RESOURCES[key].filename
        with zipfile.ZipFile(target, "w") as archive:
            for path in (source / folder).rglob("*"):
                if path.is_file():
                    relative = path.relative_to(source / ("coco-qa" if key == "cocoqa" else "ms-coco"))
                    archive.write(path, relative)
    if partially_extracted:
        shutil.copytree(source, tmp_path / "raw", dirs_exist_ok=True)
        (tmp_path / "raw/coco-qa/test/questions.txt").unlink()
    monkeypatch.chdir(tmp_path)
    assert preparation.main(["--dataset", "coco-qa", "--skip-weights"]) == 0
    assert preparation.validate_dataset(tmp_path / "data", "coco-qa")["counts"] == {"train": 1, "test": 1}


@pytest.mark.parametrize("kind", ["zip", "tar", "symlink"])
def test_archive_paths_cannot_escape_destination(tmp_path, kind):
    from mpsl.available_datasets.downloads import extract_archive
    archive = tmp_path / ("unsafe.zip" if kind == "zip" else "unsafe.tar")
    if kind == "zip":
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr("../escaped", "bad")
    else:
        with tarfile.open(archive, "w") as stream:
            info = tarfile.TarInfo("../escaped" if kind == "tar" else "link")
            if kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = "../escaped"
            else:
                info.size = 3
            stream.addfile(info, io.BytesIO(b"bad"))
    with pytest.raises(ValueError):
        extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "escaped").exists()


def test_extraction_preserves_conflicting_existing_files(tmp_path):
    from mpsl.available_datasets.downloads import extract_archive
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("image.jpg", b"original")
    out = tmp_path / "out"
    extract_archive(archive, out)
    extract_archive(archive, out)
    (out / "image.jpg").write_bytes(b"user file")
    with pytest.raises(FileExistsError):
        extract_archive(archive, out)
    assert (out / "image.jpg").read_bytes() == b"user file"


def test_kinetics_selects_only_exact_paper_filenames(tmp_path):
    from mpsl.available_datasets.downloads import extract_archive
    archive = tmp_path / "part.tar"
    with tarfile.open(archive, "w") as stream:
        for name in ("batch/clip_000001_000011.mp4", "batch/unrelated.mp4"):
            info = tarfile.TarInfo(name); info.size = 3
            stream.addfile(info, io.BytesIO(b"vid"))
    out = tmp_path / "out"
    extract_archive(archive, out, selected={"clip_000001_000011.mp4": "playing_guitar/clip_000001_000011.mp4"})
    assert (out / "playing_guitar/clip_000001_000011.mp4").read_bytes() == b"vid"
    assert len(list(out.rglob("*.mp4"))) == 1


@pytest.fixture
def http_source():
    from contextlib import contextmanager
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    @contextmanager
    def serve(payload, *, ignore_range=False, html=False, truncate=False, limited_range=False):
        ranges = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                value = self.headers.get("Range")
                ranges.append(value)
                offset = int(value.split("=")[1].split("-")[0]) if value and not ignore_range else 0
                end = min(offset + 3, len(payload)) if limited_range and value else len(payload)
                self.send_response(206 if value and not ignore_range else 200)
                self.send_header("Content-Type", "text/html" if html else "application/octet-stream")
                self.send_header("Content-Length", str(end - offset))
                self.send_header("ETag", '"one"')
                if value and not ignore_range:
                    self.send_header("Content-Range", f"bytes {offset}-{end-1}/{len(payload)}")
                self.end_headers()
                self.wfile.write(payload[offset:len(payload)//2 if truncate else end])
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/file", ranges
        finally:
            server.shutdown(); server.server_close(); worker.join()
    return serve


@pytest.mark.parametrize("ignore_range", [False, True])
def test_download_resumes_or_restarts_without_concatenating(tmp_path, http_source, ignore_range):
    from mpsl.available_datasets.downloads import download
    payload = b"complete dataset archive"
    destination = tmp_path / "archive.zip"
    with http_source(payload, ignore_range=ignore_range) as (url, ranges):
        (tmp_path / "archive.zip.part").write_bytes(payload[:5])
        (tmp_path / "archive.zip.part.json").write_text(json.dumps({"url": url, "validator": '"one"'}))
        download(url, destination)
        download(url, destination)
    assert destination.read_bytes() == payload
    assert ranges == ["bytes=5-"]
    assert not (tmp_path / "archive.zip.part").exists()


@pytest.mark.parametrize("html,truncate", [(True, False), (False, True)])
def test_download_never_publishes_error_pages_or_partial_files(tmp_path, http_source, html, truncate):
    from mpsl.available_datasets.downloads import download
    with http_source(b"not a complete dataset", html=html, truncate=truncate) as (url, _):
        with pytest.raises((ValueError, OSError)):
            download(url, tmp_path / "archive.zip", attempts=1)
    assert not (tmp_path / "archive.zip").exists()


def test_nested_meld_archives_are_prepared_without_renaming_sources(tmp_path, monkeypatch):
    from mpsl.available_datasets import acquisition
    source, samples = source_fixture(tmp_path, "meld")
    raw = tmp_path / "raw"
    raw.mkdir()
    outer = raw / acquisition.RESOURCES["meld"].filename
    with tarfile.open(outer, "w:gz") as release:
        for split, folder in (("train", "train_splits"), ("test", "output_repeated_splits_test")):
            nested = io.BytesIO()
            with tarfile.open(fileobj=nested, mode="w:gz") as archive:
                for path in (source / split).rglob("*.mkv"):
                    archive.add(path, arcname=f"{folder}/{path.name}")
            payload = nested.getvalue()
            info = tarfile.TarInfo(f"MELD.Raw/{split}.tar.gz"); info.size = len(payload)
            release.addfile(info, io.BytesIO(payload))
    monkeypatch.chdir(tmp_path)
    assert preparation.main(["--dataset", "meld", "--samples-file", str(samples), "--skip-weights"]) == 0
    assert preparation.validate_dataset(tmp_path / "data", "meld")["counts"] == {"train": 1, "test": 1}


def test_kinetics_download_layout_is_consumed_by_training_manifest(tmp_path):
    source, samples = source_fixture(tmp_path, "kinetics-sounds")
    for file in list((source / "k400/videos/train").rglob("*.mkv")):
        target = source / "kinetics-sounds/playing_accordion" / file.name
        target.parent.mkdir(parents=True, exist_ok=True)
        file.rename(target)
    manifest = preparation.prepare_dataset("kinetics-sounds", tmp_path / "data", source, samples_file=samples)
    assert manifest["counts"] == {"train": 1, "test": 1}


def test_preparation_lock_excludes_concurrent_writers(tmp_path):
    from mpsl.available_datasets.downloads import preparation_lock
    with preparation_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another preparation"):
            with preparation_lock(tmp_path):
                pytest.fail("Concurrent setup unexpectedly acquired the lock")


def test_short_range_response_is_not_a_completed_download(tmp_path, http_source):
    from mpsl.available_datasets.downloads import download
    with http_source(b"abcdefghi", limited_range=True) as (url, _):
        (tmp_path / "file.part").write_bytes(b"abc")
        (tmp_path / "file.part.json").write_text(json.dumps({"url": url, "validator": '\"one\"'}))
        with pytest.raises(OSError):
            download(url, tmp_path / "file", attempts=1)
        assert not (tmp_path / "file").exists()
        download(url, tmp_path / "file")
    assert (tmp_path / "file").read_bytes() == b"abcdefghi"


def test_kinetics_rerun_can_expand_a_custom_sample_list(tmp_path, monkeypatch):
    from mpsl.available_datasets import acquisition
    archive = tmp_path / "upstream.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name in ("a.mp4", "b.mp4"):
            info = tarfile.TarInfo(name); info.size = 3
            stream.addfile(info, io.BytesIO(b"vid"))
    index = tmp_path / "index.txt"
    index.write_text("https://s3.amazonaws.com/kinetics/400/train/part_0.tar.gz\n")
    empty = tmp_path / "empty.tar.gz"
    with tarfile.open(empty, "w:gz"):
        pass
    def offline_download(url, destination):
        destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
        origin = index if url == acquisition.KINETICS_INDEX else empty if url == acquisition.KINETICS_REPLACEMENTS else archive
        shutil.copyfile(origin, destination)
        return destination
    monkeypatch.setattr(acquisition, "download", offline_download)
    samples = tmp_path / "samples.txt"
    samples.write_text("playing_accordion/a.mp4 11 train\n")
    acquisition._kinetics(tmp_path / "raw", samples)
    samples.write_text("playing_accordion/a.mp4 11 train\nplaying_accordion/b.mp4 11 test\n")
    acquisition._kinetics(tmp_path / "raw", samples)
    assert (tmp_path / "raw/kinetics-sounds/playing_accordion/b.mp4").read_bytes() == b"vid"


def test_bad_archive_does_not_abort_other_datasets(tmp_path, monkeypatch, capsys):
    source, _ = source_fixture(tmp_path, "coco-qa")
    source.rename(tmp_path / "raw")
    (tmp_path / "raw/UCF101.rar").write_bytes(b"corrupted archive")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(preparation, "DATASETS", ("ucf101", "coco-qa"))
    assert preparation.main(["--skip-weights"]) == 1
    assert (tmp_path / "data/coco-qa/manifest.json").is_file()
    assert "[ucf101] Incomplete" in capsys.readouterr().out


@pytest.mark.parametrize("failure", ["provider", "corrupt"])
def test_encoder_failure_is_an_actionable_setup_status(tmp_path, monkeypatch, capsys, failure):
    from mpsl.available_datasets.acquisition import ENCODER_NAME
    import gdown
    from gdown.exceptions import FileURLRetrievalError
    source, _ = source_fixture(tmp_path, "coco-qa")
    source.rename(tmp_path / "raw")
    monkeypatch.chdir(tmp_path)
    if failure == "corrupt":
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "checkpoints" / ENCODER_NAME).write_bytes(b"invalid checkpoint")
    else:
        def unavailable(**kwargs):
            raise FileURLRetrievalError("Provider download limit exceeded")
        monkeypatch.setattr(gdown, "download", unavailable)
    assert preparation.main(["--dataset", "coco-qa"]) == 1
    assert "[encoder] Incomplete" in capsys.readouterr().out
    assert (tmp_path / "data/coco-qa/manifest.json").is_file()


@pytest.mark.parametrize("existing", [False, True])
def test_encoder_script_reports_invalid_local_weights_without_downloading(tmp_path, existing):
    import subprocess
    import sys
    from mpsl.available_datasets.acquisition import ENCODER_NAME
    directory = tmp_path / "checkpoints"
    if existing:
        directory.mkdir()
        (directory / ENCODER_NAME).write_bytes(b"invalid checkpoint")
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_metatransformer.py"
    result = subprocess.run(
        [sys.executable, str(script), "--validate"], cwd=tmp_path,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1
    assert "[encoder] Incomplete:" in result.stdout
    assert "Traceback" not in result.stderr
    assert directory.exists() == existing


def test_encoder_preparation_lock_is_shared_between_entrypoints(tmp_path):
    from mpsl.available_datasets.acquisition import ensure_encoder
    from mpsl.available_datasets.downloads import preparation_lock
    with preparation_lock(tmp_path, name=".encoder.lock"):
        with pytest.raises(RuntimeError, match="Another preparation"):
            ensure_encoder(tmp_path)
