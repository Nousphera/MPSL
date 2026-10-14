import copy
import threading
from argparse import Namespace
from distutils.util import strtobool

import torch

from mpsl import available_datasets as datasets
from mpsl import models
from mpsl.trainers.implementations.experiment_results import ExperimentResults
from mpsl.utils.argument_utils import validate_base_argument_constraints, build_base_argument_parser, \
    expand_argument_parser_with_adapter_approach_parameters, \
    expand_argument_parser_with_distributed_learning_parameters, set_env_variables
from mpsl.utils.config_utils import set_random_seed
from mpsl.utils.cuda_utils import get_free_cuda_device_name
from mpsl.utils.fl_utils import fed_avg, synchronize_models_in_place, get_client_weight_multipliers__nr_of_elements, \
    AGGREGATED_MODEL_NAME
from mpsl.utils.model_saving_utils import save_federated_model, save_experiment_results
from mpsl.utils.mpsl_utils import compute_mini_batch_size
from mpsl.utils.scheduler_utils import get_optimizer_and_scheduler


def init_data_iterator(client_id, client_data_iterators, client_data_loaders, barrier):
    client_data_iterators[client_id] = iter(client_data_loaders[client_id])
    barrier.wait()


def handle_train_epoch(trainer, experiment_results, device, client_data_loaders: dict, client_models: dict, epoch_nr, nr_of_clients, client_optimizers: dict, client_schedulers: dict):
    client_data_iterators = dict()

    # Note: Data iterators are initialized in separate threads to speed up loading time on slow disks with a large number of clients.
    print('Initializing train data iterators')
    barrier = threading.Barrier(nr_of_clients + 1)

    for client_id in range(nr_of_clients):
        threading.Thread(target=init_data_iterator, args=(client_id, client_data_iterators, client_data_loaders, barrier)).start()

    barrier.wait()

    return trainer.train_epoch(
        device=device,
        experiment_results=experiment_results,
        client_data_iterators=client_data_iterators,
        client_models=client_models,
        epoch_nr=epoch_nr,
        client_optimizers=client_optimizers,
        client_schedulers=client_schedulers
    )


def handle_test_epoch(trainer, experiment_results, device, client_models, dataloader, epoch_nr):
    return trainer.test_epoch(
        device=device,
        experiment_results=experiment_results,
        dataloader=dataloader,
        client_models=client_models,
        epoch_nr=epoch_nr
    )


def perform_model_aggregation_and_synchronization_in_place(_aggregated_model, _client_models: dict, client_weight_multipliers: dict):
    _aggregated_model = fed_avg(_client_models, client_weight_multipliers, aggregated_model=_aggregated_model)
    synchronize_models_in_place(_aggregated_model, _client_models)

    return _aggregated_model


def setup_arguments() -> Namespace:
    parser = build_base_argument_parser()
    parser = expand_argument_parser_with_adapter_approach_parameters(parser)
    parser = expand_argument_parser_with_distributed_learning_parameters(parser)

    parser.add_argument('--validate_only_aggregated_model', dest='validate_only_aggregated_model',
                        type=lambda x: bool(strtobool(x)), default=True,
                        help='Whether only the aggregated model -- after performing FedAvg -- should be evaluated during each epoch, rather than evaluating all client-side models.')
    parser.add_argument('--move_dist_matrix_to_cpu', dest='move_dist_matrix_to_cpu', type=lambda x: bool(strtobool(x)),
                        default=True,
                        help='Whether the computed distance matrix should be moved to the CPU prior to computing the top-k results. Used explicitly for the task of image-text retrieval.')

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
    test_dataloader = datasets.DataLoader(test_ds, batch_size=global_args.batch_size, shuffle=False, pin_memory=True, num_workers=global_args.num_workers, collate_fn=full_dataset.get_collate_fn(), drop_last=False)

    aggregated_model, trainer = models.get_federated_model_and_trainer(global_args, device)

    print(f'trainable params model: {sum(p.numel() for p in aggregated_model.parameters() if p.requires_grad)} | all: {sum(p.numel() for p in aggregated_model.parameters())}')

    client_data_loaders = dict()
    client_models = dict()

    client_optimizers = dict()
    client_schedulers = dict()

    for client_id in range(global_args.nr_of_clients):
        client_train_data = full_dataset.load_partition(partition_id=client_id)
        client_train_dataloader = datasets.DataLoader(client_train_data, batch_size=mini_batch_size, shuffle=True, pin_memory=True, num_workers=global_args.num_workers, collate_fn=full_dataset.get_collate_fn(), drop_last=False)

        client_data_loaders[client_id] = client_train_dataloader

        _client_model = copy.deepcopy(aggregated_model)
        client_models[client_id] = _client_model

        client_optimizer, client_scheduler = get_optimizer_and_scheduler(client_models[client_id], global_args)
        client_optimizers[client_id] = client_optimizer
        client_schedulers[client_id] = client_scheduler

    # A utility object that can be used to streamline archiving experiment results.
    experiment_results = ExperimentResults()

    # # # # # # # # # # # # # # # # # Training # # # # # # # # # # # # # # # # #
    for epoch_nr in range(global_args.nr_of_epochs):
        print(f'Starting epoch {epoch_nr + 1}')

        train_results_as_string, nr_of_elements_per_client_dict = handle_train_epoch(
            trainer, experiment_results, device, client_data_loaders, client_models, epoch_nr + 1, global_args.nr_of_clients, client_optimizers, client_schedulers
        )
        print(train_results_as_string)

        aggregated_model = perform_model_aggregation_and_synchronization_in_place(aggregated_model, client_models, get_client_weight_multipliers__nr_of_elements(nr_of_elements_per_client_dict))

        if global_args.save_model_after_each_epoch:
            save_federated_model(aggregated_model, client_models, global_args.save_file_name)

        models_to_evaluate = dict()
        models_to_evaluate[AGGREGATED_MODEL_NAME] = aggregated_model

        if not global_args.validate_only_aggregated_model:
            for client_id in client_models.keys():
                models_to_evaluate[client_id] = client_models[client_id]

        test_results_as_string = handle_test_epoch(
            trainer,
            experiment_results,
            device,
            models_to_evaluate,
            test_dataloader,
            epoch_nr + 1
        )
        print(test_results_as_string)

        save_experiment_results(experiment_results, global_args.save_file_name)

    print(f'Finished training - saving final model')

    if global_args.save_final_model:
        if global_args.validate_only_aggregated_model:
            # If we only validate the aggregated model, we'll also only save the aggregated model, instead of all additional client models.
            # We can enforce this by sending an empty dict as the client_models dict
            client_models = dict()

        save_federated_model(aggregated_model, client_models, global_args.save_file_name)
