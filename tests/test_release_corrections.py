import copy
import io
from types import SimpleNamespace

import pytest
import torch

from mpsl.available_datasets.multimodal.image_text_retrieval.data_partitioner import (
    image_text_retrieval__data_partitioner as partition,
)
from mpsl.models.modalities import InputModality
from mpsl.smoke import create_system, sample_inputs
from mpsl.trainers.implementations.classification import mpsl_trainer
from mpsl.utils.model_saving_utils import load_federated_model, save_federated_model


@pytest.mark.parametrize("n,k", [(10, 4), (11, 4), (3, 4), (0, 3), (29_000, 100)])
@pytest.mark.parametrize("balanced", [True, False])
def test_retrieval_partitions_cover_every_sample_once(n, k, balanced):
    parts = partition(k, n, shuffle_indices=False, use_balancing=balanced)
    assert set(parts) == set(range(k))
    assert sorted(i for values in parts.values() for i in values) == list(range(n))
    if balanced:
        assert max(map(len, parts.values())) - min(map(len, parts.values())) <= 1


def test_noncontiguous_federated_checkpoint_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_WEIGHTS_DIR", str(tmp_path))
    aggregate = torch.nn.ParameterDict({'weight': torch.nn.Parameter(torch.zeros(1))})
    clients = {i: torch.nn.ParameterDict({'weight': torch.nn.Parameter(torch.zeros(1))}) for i in (3, 7)}
    with torch.no_grad():
        for i, model in clients.items():
            model.weight.fill_(i)
    save_federated_model(aggregate, clients, "round-trip")
    with torch.no_grad():
        for model in clients.values():
            model.weight.zero_()
    _, restored = load_federated_model(aggregate, clients, "cpu", "round-trip")
    assert {i: m.weight.item() for i, m in restored.items()} == {3: 3, 7: 7}


def serialize_tensor(tensor):
    # A real wire boundary cannot preserve an autograd graph or leaf gradient.
    assert tensor.grad_fn is None and not tensor.requires_grad
    assert tensor.is_floating_point()  # Integer class labels never cross this boundary.
    buffer = io.BytesIO()
    torch.save(tensor.cpu(), buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=True).to(tensor.device)


@pytest.mark.parametrize("fusion", ["early", "late"])
def test_client_classification_loss_receives_detached_logits(fusion, monkeypatch):
    original = mpsl_trainer.loss_fn

    def client_loss(logits, labels):
        assert logits.is_leaf and logits.grad_fn is None
        return original(logits, labels)

    monkeypatch.setattr(mpsl_trainer, "loss_fn", client_loss)
    from mpsl.smoke import run

    assert run(fusion, epochs=1)["server_updated"]


@pytest.mark.parametrize("fusion", ["early", "late"])
@pytest.mark.parametrize("serialized", [False, True])
def test_explicit_logit_gradients_equal_joint_update_and_count_payload(
    fusion, serialized
):
    torch.manual_seed(17)
    clients, server = create_system(fusion, seed=17)
    reference_clients, reference_server = copy.deepcopy(clients), copy.deepcopy(server)
    xs = [sample_inputs(1), sample_inputs(2)]
    ys = [torch.tensor([0]), torch.tensor([1, 2])]
    features = [reference_clients[i](xs[i]) for i in range(2)]
    combined = (
        torch.cat(features)
        if fusion == "early"
        else {
            modality: torch.cat([f[modality] for f in features])
            for modality in features[0]
        }
    )
    torch.nn.functional.cross_entropy(
        reference_server(combined), torch.cat(ys)
    ).backward()
    for model in [*reference_clients.values(), reference_server]:
        torch.optim.SGD(model.parameters(), lr=0.1).step()

    backward_calls = []
    server.classifier.weight.register_hook(lambda grad: backward_calls.append(1))
    trainer = mpsl_trainer.MPSLTrainer(
        fusion,
        [InputModality.IMAGE, InputModality.AUDIO],
        transport=serialize_tensor if serialized else None,
    )
    method = (
        trainer._early_fusion_epoch if fusion == "early" else trainer._late_fusion_epoch
    )
    out = method(
        torch.device("cpu"),
        server,
        torch.optim.SGD(server.parameters(), lr=0.1),
        {i: iter([(xs[i], ys[i])]) for i in range(2)},
        clients,
        True,
        {i: torch.optim.SGD(m.parameters(), lr=0.1) for i, m in clients.items()},
        1,
        SimpleNamespace(nr_of_clients=2),
    )
    for actual, expected in zip(
        [*clients.values(), server], [*reference_clients.values(), reference_server]
    ):
        for a, b in zip(actual.parameters(), expected.parameters()):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)
    assert len(backward_calls) == 1
    # float32: 3*306*768 cut activations + 3*51 logit gradients + 2 scalar losses.
    assert out[2] == 2820716
    # Predictions and cut gradients travel back to clients.
    assert out[3] == 2820708
    assert sum(out[4].values()) == out[3]
    assert sum(out[5].values()) == out[2]


@pytest.mark.parametrize("serialized", [False, True])
def test_retrieval_serialized_gradients_match_local_negative_objective(
    serialized, monkeypatch
):
    from mpsl.trainers.implementations.experiment_results import ExperimentResults
    from mpsl.trainers.implementations.image_text_retrieval import (
        mpsl_trainer as retrieval,
    )
    from mpsl.trainers.implementations.image_text_retrieval.loss import (
        image_text_retrieval_loss,
    )
    from mpsl.trainers.implementations.image_text_retrieval.mpsl_trainer import (
        MPSLTrainer,
    )

    def detached_client_loss(image, text):
        assert image.is_leaf and image.grad_fn is None
        assert text._base.is_leaf and text._base.grad_fn is None
        return image_text_retrieval_loss(image, text)

    monkeypatch.setattr(retrieval, "loss_fn", detached_client_loss)

    from mpsl.models.meta_transformer.base.meta_transformer import MetaTransformerBase
    from mpsl.models.meta_transformer.implementations.multimodal.image_text_retrieval import models
    torch.manual_seed(13)
    base = MetaTransformerBase(num_classes=-1, pretrained=False, encoder_name='ViT-B/16',
                               freeze_encoder=False)
    base.apply_pre_loaded_weights()
    monkeypatch.setattr(models, 'centralized_base_model', base)
    clients = {i: torch.nn.Identity() for i in range(2)}
    server = models.ServerModel()
    reference_clients, reference_server = copy.deepcopy(clients), copy.deepcopy(server)
    batches = [(torch.randn(n, 197, 768), torch.randn(n, 2, 768)) for n in (2, 3)]
    features = [
        reference_clients[i]({InputModality.IMAGE: image, InputModality.TEXT: text})
        for i, (image, text) in enumerate(batches)
    ]
    predictions = reference_server(
        {m: torch.cat([f[m] for f in features]) for m in features[0]}
    )
    objective = 0
    for start, stop in ((0, 2), (2, 5)):
        for caption in range(2):
            loss, _ = image_text_retrieval_loss(
                predictions[InputModality.IMAGE][start:stop],
                predictions[InputModality.TEXT][start:stop, caption],
            )
            objective = objective + (stop - start) / 5 * loss
    objective.backward()
    for model in [reference_server]:
        torch.optim.SGD(model.parameters(), lr=0.1).step()
    server_backward_calls = []
    next(server.parameters()).register_hook(
        lambda grad: server_backward_calls.append(1)
    )
    trainer = MPSLTrainer(2, 5, 5, transport=serialize_tensor if serialized else None)
    optimizers = {
        i: torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1) for i in clients
    }
    results = ExperimentResults()
    trainer._train_epoch(
        results,
        server,
        torch.optim.SGD(server.parameters(), lr=0.1),
        {i: iter([batch]) for i, batch in enumerate(batches)},
        clients,
        False,
        optimizers,
        {i: torch.optim.lr_scheduler.StepLR(opt, 1) for i, opt in optimizers.items()},
        1,
        1,
        SimpleNamespace(nr_of_clients=2),
    )
    for actual, expected in zip(
        [*clients.values(), server], [*reference_clients.values(), reference_server]
    ):
        for a, b in zip(actual.parameters(), expected.parameters()):
            torch.testing.assert_close(a, b)
    assert len(server_backward_calls) == 1
    # Upload: 5*(197+2)*768*4 feature bytes, 5*3*768*4 output-gradient bytes,
    # and two scalar losses. Frozen token inputs need no returned cut gradients.
    assert results.to_json()[
        "outgoing_client_communication_overhead_in_mb"
    ] == pytest.approx(1551364 / 2**20)
    assert results.to_json()[
        "incoming_client_communication_overhead_in_mb"
    ] == pytest.approx(23040 / 2**20)
