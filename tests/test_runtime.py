import torch
import pytest
from mpsl.utils.fl_utils import fed_avg
from mpsl.utils.mpsl_utils import get_communication_size, compute_aggregated_loss
from mpsl.trainers.implementations.experiment_results import ExperimentResults


def test_communication_counts_view_payload_not_backing_storage():
    full = torch.zeros(100, 20)
    assert get_communication_size(full[:2]) == 160
    assert get_communication_size(full[:2].to(torch.float16)) == 80


def test_experiment_histories_do_not_leak_between_runs():
    first, second = ExperimentResults(), ExperimentResults()
    first.add_results(1, 0.75, False)
    assert second.to_json()["train_metric"] == []
    assert second.to_json()["epochs"] == []


def test_fedavg_accepts_actual_participating_client_ids():
    a, b = torch.nn.Linear(2, 1, bias=False), torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        a.weight.fill_(1)
        b.weight.fill_(3)
    result = fed_avg({3: a, 7: b}, {3: 0.25, 7: 0.75})
    torch.testing.assert_close(result.weight, torch.full((1, 2), 2.5))
    torch.testing.assert_close(a.weight, torch.ones(1, 2))


def test_weighted_loss_retains_correct_gradients():
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    loss = compute_aggregated_loss([(x * x, 1), (y * y, 3)], 4)
    loss.backward()
    assert x.grad.item() == pytest.approx(1.0)
    assert y.grad.item() == pytest.approx(4.5)


def test_split_gradient_matches_joint_weighted_batch():
    import copy
    from types import SimpleNamespace
    from mpsl.trainers.implementations.classification.mpsl_trainer import MPSLTrainer

    torch.manual_seed(4)
    clients = {i: torch.nn.Linear(2, 3) for i in range(2)}
    server = torch.nn.Linear(3, 2)
    reference_clients = copy.deepcopy(clients)
    reference_server = copy.deepcopy(server)
    xs = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[2.0, 1.0]])]
    ys = [torch.tensor([0, 1]), torch.tensor([1])]
    ref_opts = [
        torch.optim.SGD(m.parameters(), lr=0.1)
        for m in [*reference_clients.values(), reference_server]
    ]
    logits = reference_server(
        torch.cat([reference_clients[i](xs[i]) for i in range(2)])
    )
    torch.nn.functional.cross_entropy(logits, torch.cat(ys)).backward()
    for opt in ref_opts:
        opt.step()
    trainer = MPSLTrainer("early", [])
    trainer._early_fusion_epoch(
        torch.device("cpu"),
        server,
        torch.optim.SGD(server.parameters(), lr=0.1),
        {i: iter([(xs[i], ys[i])]) for i in range(2)},
        clients,
        True,
        {i: torch.optim.SGD(clients[i].parameters(), lr=0.1) for i in range(2)},
        1,
        SimpleNamespace(nr_of_clients=2),
    )
    for actual, expected in zip(
        [*clients.values(), server], [*reference_clients.values(), reference_server]
    ):
        for a, b in zip(actual.parameters(), expected.parameters()):
            torch.testing.assert_close(a, b)


def test_evaluation_weights_partial_batches_by_examples():
    from torch.utils.data import TensorDataset, DataLoader
    from mpsl.trainers.implementations.classification.mpsl_trainer import MPSLTrainer

    results = ExperimentResults()
    data = TensorDataset(
        torch.tensor([[2.0, 0.0]]).repeat(5, 1), torch.tensor([0, 0, 0, 0, 1])
    )
    MPSLTrainer("early", [])._test_epoch(
        torch.nn.Identity(),
        torch.nn.Identity(),
        DataLoader(data, batch_size=4),
        torch.device("cpu"),
        results,
        1,
    )
    assert results.test_metric[-1] == pytest.approx(0.8)


def test_dataset_registry_is_lazy_and_partition_ids_cover_data():
    from mpsl import available_datasets

    assert "coco-qa" in available_datasets.available_datasets()

    class Dataset(torch.utils.data.Dataset):
        targets = [0] * 20 + [1] * 20

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return index, self.targets[index]

    partitioner = available_datasets.DirichletDataPartitioner(
        Dataset(), 2, 1.0, min_partition_size=1, seed=8
    )
    parts = [partitioner.load_partition(i).indices for i in range(2)]
    assert sorted(parts[0] + parts[1]) == list(range(40))


def test_model_payload_accounts_for_dtype():
    from mpsl.utils.fl_utils import get_communication_size_for_model_in_bytes

    assert get_communication_size_for_model_in_bytes(torch.nn.Linear(2, 3).half()) == 18


@pytest.mark.parametrize("fusion", ["early", "late"])
def test_real_trainer_smoke_is_deterministic_and_updates_server(fusion):
    from mpsl.smoke import run

    first, second = run(fusion, epochs=2), run(fusion, epochs=2)
    assert first == second
    assert first["server_updated"]
    assert first["history"][-1]["loss"] < first["history"][0]["loss"]
    assert all(
        row["server_incoming_bytes"] > 0 and row["server_outgoing_bytes"] > 0
        for row in first["history"]
    )


@pytest.mark.parametrize("kind", ["centralized", "federated"])
def test_baseline_evaluation_weights_partial_batches(kind):
    from torch.utils.data import DataLoader, TensorDataset

    data = DataLoader(
        TensorDataset(
            torch.tensor([[2.0, 0.0]]).repeat(5, 1), torch.tensor([0, 0, 0, 0, 1])
        ),
        batch_size=4,
    )
    results = ExperimentResults()
    if kind == "centralized":
        from mpsl.trainers.implementations.classification.centralized_trainer import (
            CentralizedTrainer,
        )

        CentralizedTrainer()._perform_epoch(
            results, torch.nn.Identity(), torch.device("cpu"), data, 1, None
        )
    else:
        from mpsl.trainers.implementations.classification.fl_trainer import FLTrainer

        class Model(torch.nn.Identity):
            def switch_to_device(self, device):
                return self.to(device)

        FLTrainer()._perform_epoch(
            torch.device("cpu"),
            results,
            {0: iter(data)},
            {0: Model()},
            1,
            dataloader_length=len(data),
        )
    assert results.test_metric[-1] == pytest.approx(0.8)
