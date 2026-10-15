"""Startup checks that do not download data or pretrained checkpoints."""

import importlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from torch.utils.data import DataLoader

from mpsl.utils.argument_utils import build_base_argument_parser, set_env_variables


def epoch_arguments(kind, loaders, trainer):
    arguments = dict(
        trainer=trainer,
        experiment_results=None,
        device="cpu",
        client_models={},
        client_optimizers={},
        client_schedulers={},
        epoch_nr=1,
    )
    if kind == "mpsl":
        arguments.update(
            server_model=None,
            server_optimizer=None,
            server_scheduler=None,
            client_dataloaders=loaders,
            client_model_requires_any_grad=False,
            max_nr_of_batches_in_epoch=1,
            global_args=SimpleNamespace(nr_of_clients=len(loaders)),
        )
    else:
        arguments.update(client_data_loaders=loaders, nr_of_clients=len(loaders))
    return arguments


@pytest.mark.parametrize("kind", ["mpsl", "fl"])
def test_training_starts_with_each_clients_data_iterator(kind):
    module = importlib.import_module(f"mpsl.main_{kind}")
    loaders = {0: DataLoader([11, 12], batch_size=2), 1: DataLoader([21])}

    class Trainer:
        def train_epoch(self, *, client_data_iterators, **kwargs):
            return {
                client: next(iterator).tolist()
                for client, iterator in client_data_iterators.items()
            }

    assert module.handle_train_epoch(**epoch_arguments(kind, loaders, Trainer())) == {
        0: [11, 12],
        1: [21],
    }


@pytest.mark.parametrize("kind", ["mpsl", "fl"])
def test_loader_startup_failure_reaches_the_training_caller(kind):
    # A subprocess bounds a regression to the original barrier deadlock.
    script = f"""
import importlib
from test_training_entrypoints import epoch_arguments

class FailedLoader:
    def __iter__(self):
        raise OSError('dataset worker failed to start')

module = importlib.import_module('mpsl.main_{kind}')
try:
    module.handle_train_epoch(**epoch_arguments('{kind}', {{0: [1], 1: FailedLoader()}}, None))
except OSError as error:
    assert str(error) == 'dataset worker failed to start'
else:
    raise AssertionError('The caller did not receive the loader error')
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).parent), environment.get("PYTHONPATH", "")]
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True,
            timeout=10, env=environment,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("A DataLoader initialization error left training waiting forever")
    assert result.returncode == 0, result.stderr


def test_federated_entrypoint_does_not_require_distutils(monkeypatch):
    monkeypatch.setitem(sys.modules, "distutils.util", None)
    module = importlib.import_module("mpsl.main_fl")
    importlib.reload(module)


def test_default_paths_create_local_run_directories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key in (
        "TORCH_DATA_DIR", "PRE_PROCESSORS_CACHE_DIR", "TOKENIZER_WEIGHTS_CACHE_DIR",
        "MODEL_WEIGHTS_DIR", "VIT_BASE",
    ):
        monkeypatch.setenv(key, "previous-value")
    args = build_base_argument_parser().parse_args(
        ["--dataset", "coco-qa", "--save_file_name", "run"]
    )
    set_env_variables(args)
    for key, relative_path in {
        "TORCH_DATA_DIR": "data",
        "PRE_PROCESSORS_CACHE_DIR": "cache/preprocessors",
        "TOKENIZER_WEIGHTS_CACHE_DIR": "cache/tokenizers",
        "MODEL_WEIGHTS_DIR": "checkpoints",
    }.items():
        assert Path(os.environ[key]).resolve() == tmp_path / relative_path
        assert (tmp_path / relative_path).is_dir()


def test_explicit_paths_are_created_without_overwriting_files(tmp_path, monkeypatch):
    existing = tmp_path / "existing-cache"
    existing.write_text("keep this file")
    args = build_base_argument_parser().parse_args(
        [
            "--dataset", "coco-qa", "--save_file_name", "run",
            "--torch_data_dir", str(tmp_path / "custom-data"),
            "--pre_processors_cache_dir", str(existing),
            "--tokenizer_weights_cache_dir", str(tmp_path / "custom-tokenizers"),
            "--model_weights_dir", str(tmp_path / "custom-checkpoints"),
        ]
    )
    monkeypatch.setenv("MODEL_WEIGHTS_DIR", "previous-value")
    with pytest.raises(FileExistsError):
        set_env_variables(args)
    assert existing.read_text() == "keep this file"
    assert os.environ["MODEL_WEIGHTS_DIR"] == "previous-value"
