"""Offline regressions for training configuration and dataset dispatch."""

import pytest
import torch
from torch.utils.data import DataLoader

from mpsl import models
from mpsl.trainers.implementations.experiment_results import ExperimentResults
from mpsl.trainers.implementations.image_text_retrieval.fl_trainer import FLTrainer
from mpsl.utils.argument_utils import (
    build_base_argument_parser,
    expand_argument_parser_with_adapter_approach_parameters,
    expand_argument_parser_with_distributed_learning_parameters,
    validate_base_argument_constraints,
)


@pytest.mark.parametrize("kind", ["centralized", "federated"])
@pytest.mark.parametrize("adapter", ["audio", "text", "unified"])
def test_meld_dispatch_builds_requested_adapter(kind, adapter, monkeypatch):
    from mpsl.models.meta_transformer.implementations.multimodal.emotion_recognition import (
        models as meld,
    )

    # Keep the real factory and adapter construction; avoid external weights
    # and the full encoder, which are unrelated to option dispatch.
    def initialize_base(self, *, num_classes, **kwargs):
        torch.nn.Module.__init__(self)
        self.embed_dim = 4
        self.encoder = torch.nn.Linear(4, 4)
        self.classifier = torch.nn.Linear(4, num_classes)

    monkeypatch.setattr(meld.CentralizedModel, "__init__", initialize_base)
    monkeypatch.setattr(meld, "centralized_base_model", None)
    monkeypatch.setattr(models, "_prepared_class_count", lambda dataset: 7)
    parser = expand_argument_parser_with_adapter_approach_parameters(
        build_base_argument_parser()
    )
    args = parser.parse_args(
        ["--dataset", "meld", "--save_file_name", "test",
         "--use_adapter_approach", "True", f"--include_{adapter}_adapter", "True"]
    )
    model, _ = getattr(models, f"get_{kind}_model_and_trainer")(args, "cpu")
    assert isinstance(model, meld.CentralizedModelWithAdapter)
    assert all(p.requires_grad for p in getattr(model, f"{adapter}_adapter").parameters())
    assert not any(p.requires_grad for p in model.encoder.parameters())


class RetrievalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(1.0)

    def switch_to_device(self, device):
        return self.to(device)

    def forward(self, inputs):
        return {modality: self.dropout(value) for modality, value in inputs.items()}


def retrieval_loader():
    images = torch.eye(2)
    return DataLoader([(image, image.unsqueeze(0)) for image in images], batch_size=2)


def test_retrieval_evaluation_disables_training_layers():
    results = ExperimentResults()
    FLTrainer(1, 2, 2, k_vals=[1])._test_epoch(
        "cpu", results, retrieval_loader(), {"aggregated_model": RetrievalModel()}, 1
    )
    assert results.test_metric == [[[100.0], [100.0]]]


@pytest.mark.parametrize("aggregate_first", [True, False])
def test_retrieval_records_aggregate_when_also_evaluating_clients(aggregate_first):
    ids = ["aggregated_model", 0] if aggregate_first else [0, "aggregated_model"]
    clients = {client_id: RetrievalModel().eval() for client_id in ids}
    results = ExperimentResults()
    FLTrainer(1, 2, 2, k_vals=[1])._test_epoch(
        "cpu", results, retrieval_loader(), clients, 1
    )
    assert results.test_metric == [[[100.0], [100.0]]]


def training_arguments(**changes):
    parser = expand_argument_parser_with_distributed_learning_parameters(
        build_base_argument_parser()
    )
    parser.add_argument("--test_num_workers", type=int, default=0)
    args = parser.parse_args(
        ["--dataset", "coco-qa", "--save_file_name", "test", "--batch_size", "500",
         "--nr_of_clients", "100", "--dataset_split_type", "noniid"]
    )
    for key, value in changes.items():
        setattr(args, key, value)
    return args


@pytest.mark.parametrize("option,value", [
    ("batch_size", 0), ("batch_size", -1),
    ("nr_of_epochs", 0), ("nr_of_epochs", -1),
    ("scheduler_step_size", 0), ("scheduler_step_size", -1),
    ("nr_of_clients", 0), ("nr_of_clients", -1),
    ("start_lr", -1), ("start_lr", float("nan")), ("start_lr", float("inf")),
    ("dirichlet_alpha", 0), ("dirichlet_alpha", -1),
    ("dirichlet_alpha", float("nan")), ("dirichlet_alpha", float("inf")),
    ("num_workers", -1), ("test_num_workers", -1),
])
def test_invalid_training_numbers_fail_before_data_loading(option, value):
    with pytest.raises(ValueError, match=option):
        validate_base_argument_constraints(training_arguments(**{option: value}))


@pytest.mark.parametrize("dataset,audio", [("coco-qa", None), ("ucf101", "1s"), ("meld", "3.88s")])
def test_valid_paper_settings_and_zero_learning_rate_are_accepted(dataset, audio):
    args = training_arguments(
        dataset=dataset, audio_time_dimension=audio, start_lr=0,
        nr_of_last_encoder_blocks_to_finetune=6, num_workers=0,
    )
    validate_base_argument_constraints(args)
    assert args.start_lr == 0
