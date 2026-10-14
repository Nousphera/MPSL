from argparse import Namespace

import torch

from mpsl import available_datasets as datasets
from mpsl.models import get_centralized_model_and_trainer
from mpsl.trainers.implementations.experiment_results import ExperimentResults
from mpsl.utils.argument_utils import build_base_argument_parser, validate_base_argument_constraints, \
    expand_argument_parser_with_adapter_approach_parameters, set_env_variables
from mpsl.utils.config_utils import set_random_seed
from mpsl.utils.cuda_utils import get_free_cuda_device_name
from mpsl.utils.model_saving_utils import save_centralized_model, \
    save_experiment_results
from mpsl.utils.scheduler_utils import get_optimizer_and_scheduler


def handle_train_epoch(trainer, experiment_results, model, device, dataloader, epoch_nr, optimizer):
    return trainer.train_epoch(
        experiment_results=experiment_results,
        model=model,
        device=device,
        dataloader=dataloader,
        epoch_nr=epoch_nr,
        optimizer=optimizer
    )


def handle_test_epoch(trainer, experiment_results, model, device, dataloader, epoch_nr):
    return trainer.test_epoch(
        experiment_results=experiment_results,
        model=model,
        device=device,
        dataloader=dataloader,
        epoch_nr=epoch_nr
    )


def setup_arguments() -> Namespace:
    parser = build_base_argument_parser()
    parser = expand_argument_parser_with_adapter_approach_parameters(parser)

    args: Namespace = parser.parse_args()

    validate_base_argument_constraints(args)
    set_env_variables(args)

    return args


if __name__ == '__main__':
    global_args = setup_arguments()
    print(global_args)

    set_random_seed(global_args.random_seed)

    # # # # # # # # # # # # # # # # # Setup # # # # # # # # # # # # # # # # #
    device = torch.device(get_free_cuda_device_name(global_args) if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    full_dataset = datasets.load_data(name=global_args.dataset, num_partitions=1, split='iid', seed=global_args.random_seed, global_args=global_args)
    train_data = full_dataset.load_partition(partition_id=0)
    train_dataloader = datasets.DataLoader(train_data, batch_size=global_args.batch_size, shuffle=True, pin_memory=True, num_workers=global_args.num_workers, collate_fn=full_dataset.get_collate_fn())

    test_ds = full_dataset.load_test_set()
    test_dataloader = datasets.DataLoader(test_ds, batch_size=global_args.batch_size, shuffle=False, pin_memory=True, num_workers=global_args.num_workers, collate_fn=full_dataset.get_collate_fn())

    full_model, trainer = get_centralized_model_and_trainer(global_args, device)
    full_model = full_model.switch_to_device(device)

    print(f'trainable params centralized model: {sum(p.numel() for p in full_model.parameters() if p.requires_grad)} | all: {sum(p.numel() for p in full_model.parameters())}')

    optimizer, scheduler = get_optimizer_and_scheduler(full_model, global_args)

    # A utility object that can be used to streamline archiving experiment results.
    experiment_results = ExperimentResults()

    # # # # # # # # # # # # # # # # # Training # # # # # # # # # # # # # # # # #
    for epoch_nr in range(global_args.nr_of_epochs):
        print(f'Starting epoch {epoch_nr + 1}')

        train_results_as_string = handle_train_epoch(trainer, experiment_results, full_model, device, train_dataloader, epoch_nr + 1, optimizer)
        print(train_results_as_string)

        scheduler.step()

        if global_args.save_model_after_each_epoch:
            save_centralized_model(full_model, global_args.save_file_name)

        with torch.no_grad():
            print('Starting evaluation')
            test_results_as_string = handle_test_epoch(trainer, experiment_results, full_model, device, test_dataloader, epoch_nr + 1)

            print(test_results_as_string)

        save_experiment_results(experiment_results, global_args.save_file_name)

    print(f'Finished training - Saving final model')

    if global_args.save_final_model:
        save_centralized_model(full_model, global_args.save_file_name)

