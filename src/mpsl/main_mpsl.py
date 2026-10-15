import copy
import threading
from argparse import Namespace

import torch

from mpsl import available_datasets as datasets
from mpsl import models
from mpsl.trainers.implementations.experiment_results import ExperimentResults
from mpsl.trainers.implementations.experiment_trainer import ExperimentTrainer
from mpsl.utils.argument_utils import build_base_argument_parser, \
    expand_argument_parser_with_distributed_learning_parameters, validate_base_argument_constraints, set_env_variables
from mpsl.utils.config_utils import set_random_seed
from mpsl.utils.cuda_utils import get_free_cuda_device_name
from mpsl.utils.fl_utils import fed_avg, get_client_weight_multipliers__nr_of_elements
from mpsl.utils.model_saving_utils import save_split_model, save_experiment_results
from mpsl.utils.mpsl_utils import compute_mini_batch_size
from mpsl.utils.scheduler_utils import get_optimizer_and_scheduler


def init_data_iterator(client_id, client_data_iterators, client_dataloaders, barrier):
    client_data_iterators[client_id] = iter(client_dataloaders[client_id])
    barrier.wait()


def handle_train_epoch(
        trainer: ExperimentTrainer,
        experiment_results,
        device,
        server_model,
        server_optimizer,
        server_scheduler,
        client_dataloaders,
        client_models,
        client_model_requires_any_grad,
        max_nr_of_batches_in_epoch,
        client_optimizers,
        client_schedulers,
        epoch_nr,
        global_args):
    client_data_iterators = dict()

    # Note: Data iterators are initialized in separate threads to speed up loading time on slow disks with a large number of clients.
    print('Initializing train data iterators')
    barrier = threading.Barrier(global_args.nr_of_clients + 1)

    for client_id in range(global_args.nr_of_clients):
        threading.Thread(target=init_data_iterator, args=(client_id, client_data_iterators, client_dataloaders, barrier)).start()

    barrier.wait()

    return trainer.train_epoch(
        device=device,
        experiment_results=experiment_results,
        server_model=server_model,
        server_optimizer=server_optimizer,
        server_scheduler=server_scheduler,
        client_data_iterators=client_data_iterators,
        client_models=client_models,
        client_model_requires_any_grad=client_model_requires_any_grad,
        client_optimizers=client_optimizers,
        client_schedulers=client_schedulers,
        max_nr_of_batches_in_epoch=max_nr_of_batches_in_epoch,
        epoch_nr=epoch_nr,
        global_args=global_args
    )


def handle_test_epoch(trainer, experiment_results, device, client_model, server_model, dataloader, epoch_nr):
    return trainer.test_epoch(
        device=device,
        experiment_results=experiment_results,
        server_model=server_model,
        client_model=client_model,
        dataloader=dataloader,
        epoch_nr=epoch_nr
    )


def setup_arguments() -> Namespace:
    parser = build_base_argument_parser()
    parser = expand_argument_parser_with_distributed_learning_parameters(parser)

    parser.add_argument('--test_num_workers', type=int, default=5,
                        help='num_workers provided to the test Dataloader. For Split Learning, we differentiate between num_workers for the train Dataloader and test_num_workers for the test Dataloader.')

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
    mini_batch_size = compute_mini_batch_size(global_args.batch_size, global_args.nr_of_clients)
    total_batch_size = global_args.nr_of_clients * mini_batch_size

    print(f'Device: {device}')
    print(f'Available datasets: {datasets.available_datasets()}, chosen: {global_args.dataset} with batch_size: {global_args.batch_size} and mini-batch size: {mini_batch_size}. Hence real total batch_size is {total_batch_size}')

    if total_batch_size > global_args.batch_size:
        raise Exception(f'total_batch_size > chosen batch_size ({total_batch_size} > {global_args.batch_size})')

    full_dataset = datasets.load_data(name=global_args.dataset, num_partitions=global_args.nr_of_clients, split=global_args.dataset_split_type, seed=global_args.random_seed, global_args=global_args)
    test_ds = full_dataset.load_test_set()
    test_dataloader = datasets.DataLoader(test_ds, batch_size=global_args.batch_size, shuffle=False, pin_memory=True, num_workers=global_args.test_num_workers, collate_fn=full_dataset.get_collate_fn(), drop_last=False)

    (client_model, server_model, client_model_requires_any_grad), trainer = models.get_split_model_pair_and_trainer(global_args, device)
    server_model = server_model.switch_to_device(device)

    client_dataloaders = dict()
    client_models = dict()

    client_optimizers = dict()
    client_schedulers = dict()

    max_nr_of_batches_in_epoch = 0

    print(f'Number of trainable params server model: {sum(p.numel() for p in server_model.parameters() if p.requires_grad)} | Total number of parameters: {sum(p.numel() for p in server_model.parameters())}')
    print(f'Number of trainable params client model: {sum(p.numel() for p in client_model.parameters() if p.requires_grad)} | Total number of parameters: {sum(p.numel() for p in client_model.parameters())}')

    for client_id in range(global_args.nr_of_clients):
        client_train_data = full_dataset.load_partition(partition_id=client_id)
        client_train_dataloader = datasets.DataLoader(client_train_data, batch_size=mini_batch_size, shuffle=True, pin_memory=True, num_workers=global_args.num_workers, collate_fn=full_dataset.get_collate_fn(), drop_last=False)

        client_dataloaders[client_id] = client_train_dataloader

        _client_model = client_model if client_id == 0 else copy.deepcopy(client_model)
        _client_model = _client_model.switch_to_device(device)
        client_optimizer, client_scheduler = get_optimizer_and_scheduler(_client_model, global_args)
        client_optimizers[client_id] = client_optimizer
        client_models[client_id] = _client_model

        if client_model_requires_any_grad:
            client_schedulers[client_id] = client_scheduler

        max_nr_of_batches_in_epoch = max(max_nr_of_batches_in_epoch, len(client_train_dataloader))

    server_optimizer, server_scheduler = get_optimizer_and_scheduler(server_model, global_args)

    # A utility object that can be used to streamline archiving experiment results.
    experiment_results = ExperimentResults(global_args)

    # # # # # # # # # # # # # # # # # Training # # # # # # # # # # # # # # # # #
    for epoch_nr in range(global_args.nr_of_epochs):
        print(f'Starting epoch {epoch_nr + 1}')

        train_results_as_string, nr_of_elements_per_client_dict = handle_train_epoch(
            trainer,
            experiment_results,
            device,
            server_model,
            server_optimizer,
            server_scheduler,
            client_dataloaders,
            client_models,
            client_model_requires_any_grad,
            max_nr_of_batches_in_epoch,
            client_optimizers,
            client_schedulers,
            epoch_nr + 1,
            global_args
        )
        print(train_results_as_string)

        server_scheduler.step()

        aggregated_client_model = fed_avg(client_models, get_client_weight_multipliers__nr_of_elements(nr_of_elements_per_client_dict))

        if global_args.save_model_after_each_epoch:
            save_split_model(aggregated_client_model, server_model, global_args.save_file_name)

        test_results_as_string = handle_test_epoch(
            trainer,
            experiment_results,
            device,
            aggregated_client_model,
            server_model,
            test_dataloader,
            epoch_nr + 1
        )
        print(test_results_as_string)

        save_experiment_results(experiment_results, global_args.save_file_name)

    print(f'Finished training - saving final model')

    if global_args.save_final_model:
        save_split_model(aggregated_client_model, server_model, global_args.save_file_name)
