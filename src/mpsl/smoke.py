"""Offline split training with the paper Meta-Transformer ViT-B/16 architecture.

Uses the UCF101 audio-subset model (51 classes), 224px images, and the supported
one-second AST patch tokenizer. All weights are explicitly initialized from
scratch; only sample/step counts are reduced. No pretrained fallback is used.
"""

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from mpsl.models.modalities import InputModality
from mpsl.models.meta_transformer.implementations.multimodal.action_recognition.models import (
    CentralizedModel,
    ClientModel,
    ServerModel,
)
from mpsl.trainers.implementations.classification.mpsl_trainer import MPSLTrainer
from mpsl.trainers.implementations.experiment_results import ExperimentResults
from mpsl.utils.fl_utils import fed_avg


def create_system(fusion="early", seed=7):
    if fusion not in {"early", "late"}:
        raise ValueError("Use early or late fusion")
    torch.manual_seed(seed)
    base = CentralizedModel(
        use_pre_layer_norm=False,
        use_post_layer_norm=False,
        use_large_encoder=False,
        freeze_encoder=False,
        nr_of_encoder_blocks_to_finetune=6,
        trainable_params_key="default",
        device="cpu",
        fusion_type=f"{fusion}_with_both_cls_and_pooling_mean",
        audio_time_dimension="1s",
        num_classes=51,
        pretrained=False,
        encoder_name="ViT-B/16",
    )
    client = ClientModel("cpu", base)
    return {0: client, 1: copy.deepcopy(client)}, ServerModel(base)


def sample_inputs(count):
    return {
        InputModality.IMAGE: torch.randn(count, 3, 224, 224),
        InputModality.AUDIO: torch.randn(count, 128, 98),
    }


def run(fusion="early", epochs=1, seed=7):
    if epochs < 1:
        raise ValueError("Use at least one epoch")
    torch.set_num_threads(1)
    clients, server = create_system(fusion, seed)
    before = server.classifier.weight.detach().clone()
    # Client 1 has an extra sample, exercising batch exhaustion and weighting.
    data = {
        i: [(sample_inputs(1), torch.tensor([j])) for j in range(1 + i)]
        for i in clients
    }
    optimizers = {
        i: torch.optim.Adam(m.parameters(), lr=1e-4) for i, m in clients.items()
    }
    server_optimizer = torch.optim.Adam(server.parameters(), lr=1e-4)
    trainer = MPSLTrainer(fusion, [InputModality.IMAGE, InputModality.AUDIO])
    results, history = ExperimentResults(), []
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
            {i: iter(batches) for i, batches in data.items()},
            clients,
            True,
            optimizers,
            2,
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
        averaged = fed_avg(clients, {0: 1 / 3, 1: 2 / 3})
        trainer._test_epoch(
            averaged,
            server,
            [(sample_inputs(1), torch.tensor([0]))],
            torch.device("cpu"),
            results,
            epoch,
        )
    return {
        "scope": "paper architecture execution check; synthetic inputs",
        "architecture": "Meta-Transformer ViT-B/16",
        "initialization": "from_scratch",
        "fusion": fusion,
        "seed": seed,
        "history": history,
        "test_accuracy": results.test_metric,
        "server_updated": not torch.equal(before, server.classifier.weight),
        "partitions": [[0], [1, 2]],
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
