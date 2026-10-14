import os

import torch

from mpsl.trainers.implementations.experiment_results import ExperimentResults
from mpsl.utils.file_utils import save_object_to_json
from mpsl.utils.fl_utils import AGGREGATED_MODEL_NAME
from mpsl.utils.mpsl_utils import get_client_name


def _save_object(obj, file_name):
    if file_name is None or len(file_name) == 0:
        raise Exception('file_name should not be None or an empty string.')

    file_name = f'{file_name}.pth'

    os.makedirs(os.environ['MODEL_WEIGHTS_DIR'], exist_ok=True)
    torch.save(obj, os.path.join(os.environ['MODEL_WEIGHTS_DIR'], file_name))
    print(f'Saved model in filename: {file_name}')


def _load_checkpoint(file_name, device):
    if file_name is None or len(file_name) == 0:
        raise Exception('file_name should not be None or an empty string.')

    file_name = f'{file_name}.pth'
    checkpoint = torch.load(os.path.join(os.environ['MODEL_WEIGHTS_DIR'], file_name), map_location=device, weights_only=True)

    return checkpoint


def save_centralized_model(centralized_model, desired_file_name):
    _save_object(centralized_model.state_dict(), desired_file_name)


def save_split_model(client_model, server_model, desired_file_name):
    _save_object(
        {
            'client_model': client_model.state_dict(),
            'server_model': server_model.state_dict()
        },
        desired_file_name
    )


def save_federated_model(aggregated_model, client_models: dict, desired_file_name):
    save_object = {
            AGGREGATED_MODEL_NAME: aggregated_model.state_dict(),
    }

    for client_id in client_models.keys():
        save_object[get_client_name(client_id)] = client_models[client_id].state_dict()

    _save_object(
        save_object,
        desired_file_name
    )


def load_centralized_model(centralized_model, device, checkpoint_file_name):
    checkpoint = _load_checkpoint(checkpoint_file_name, device)

    centralized_model.load_state_dict(checkpoint)

    return centralized_model


def load_split_model(client_model, server_model, device, checkpoint_file_name):
    checkpoint = _load_checkpoint(checkpoint_file_name, device)

    if client_model is not None:
        client_model.load_state_dict(checkpoint['client_model'])
    server_model.load_state_dict(checkpoint['server_model'])

    return client_model, server_model


def load_federated_model(aggregated_model, client_models: dict, device, checkpoint_file_name):
    """
    :param client_models: A dict consisting of instances of the client models, into which the checkpoint will be loaded.
    :param aggregated_model: An instance of the aggregated_model model, into which the checkpoint will be loaded.
    :return: a tuple of the form (aggregated_model, dict<client_models>) initialized with the desired weights.
    """
    checkpoint = _load_checkpoint(checkpoint_file_name, device)

    aggregated_model.load_state_dict(checkpoint[AGGREGATED_MODEL_NAME])

    for client_id, model in client_models.items():
        model.load_state_dict(checkpoint[get_client_name(client_id)])

    return aggregated_model, client_models


def save_experiment_results(experiment_results: ExperimentResults, desired_file_name):
    """
    Attempts to save the results object that is a json object of all results (to provide more ease of use).
    """
    if desired_file_name is None:
        raise Exception('desired_file_name was None: cannot save results dict')
    else:
        full_file_path = os.path.join(os.environ['MODEL_WEIGHTS_DIR'], desired_file_name + '.json')

        save_object_to_json(experiment_results.to_json(), full_file_path)
