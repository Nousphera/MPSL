"""Offline execution check of the split-learning trainer.

Two synthetic modalities and small trainable networks exercise both fusion paths
without model or dataset downloads.
"""

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mpsl.models.modalities import InputModality
from mpsl.trainers.implementations.classification.mpsl_trainer import MPSLTrainer
from mpsl.trainers.implementations.experiment_results import ExperimentResults
from mpsl.utils.fl_utils import fed_avg


class Client(nn.Module):
    def __init__(self, fusion):
        super().__init__()
        self.fusion = fusion
        self.image = nn.Linear(4, 6)
        self.audio = nn.Linear(4, 6)

    def forward(self, x):
        image, audio = self.image(x[:, :4]), self.audio(x[:, 4:])
        if self.fusion == "early":
            return torch.cat((image, audio), dim=1)
        return {InputModality.IMAGE: image, InputModality.AUDIO: audio}


class Server(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(12, 12), nn.GELU(), nn.Linear(12, 2))

    def forward(self, x):
        if isinstance(x, dict):
            x = torch.cat((x[InputModality.IMAGE], x[InputModality.AUDIO]), dim=1)
        return self.classifier(x)


def run(fusion="early", epochs=3, seed=7):
    if fusion not in {"early", "late"} or epochs < 1:
        raise ValueError("Use early/late fusion and at least one epoch")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    x = torch.randn(60, 8)
    y = (x[:, 0] + x[:, 4] > 0).long()
    # Uneven client and batch sizes deliberately exercise weighting and exhaustion.
    indices = [list(range(0, 21)), list(range(21, 45))]
    loaders = {
        i: DataLoader(TensorDataset(x[part], y[part]), batch_size=8)
        for i, part in enumerate(indices)
    }
    clients = {i: Client(fusion) for i in range(2)}
    server = Server()
    before = copy.deepcopy(server.state_dict())
    optimizers = {
        i: torch.optim.SGD(model.parameters(), lr=0.1) for i, model in clients.items()
    }
    server_optimizer = torch.optim.SGD(server.parameters(), lr=0.1)
    trainer = MPSLTrainer(fusion, [InputModality.IMAGE, InputModality.AUDIO])
    results = ExperimentResults()
    history = []
    for epoch in range(1, epochs + 1):
        for model in [*clients.values(), server]:
            model.train()
        method = (
            trainer._early_fusion_epoch
            if fusion == "early"
            else trainer._late_fusion_epoch
        )
        out = method(
            torch.device("cpu"),
            server,
            server_optimizer,
            {i: iter(loader) for i, loader in loaders.items()},
            clients,
            True,
            optimizers,
            max(map(len, loaders.values())),
            SimpleNamespace(nr_of_clients=2),
        )
        total = sum(out[-1].values())
        history.append(
            {
                "epoch": epoch,
                "loss": out[0] / total,
                "accuracy": out[1] / total,
                "server_incoming_bytes": out[2],
                "server_outgoing_bytes": out[3],
            }
        )
        averaged = fed_avg(clients, {0: 21 / 45, 1: 24 / 45})
        trainer._test_epoch(
            averaged,
            server,
            DataLoader(TensorDataset(x[45:], y[45:]), batch_size=8),
            torch.device("cpu"),
            results,
            epoch,
        )
    return {
        "scope": "synthetic execution check",
        "fusion": fusion,
        "seed": seed,
        "history": history,
        "test_accuracy": results.test_metric,
        "server_updated": any(
            not torch.equal(before[k], v) for k, v in server.state_dict().items()
        ),
        "partitions": indices,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/smoke"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    result = {fusion: run(fusion) for fusion in ("early", "late")}
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
