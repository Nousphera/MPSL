import copy
import torch

NR_OF_BYTES_PER_FLOAT32 = 4
AGGREGATED_MODEL_NAME = 'aggregated_model'


def fed_avg(models: dict, client_weight_multipliers: dict, aggregated_model=None):
    # Taking any arbitrary client_model as the base will do, as all parameters will be modified anyways. We'll resort to simply choosing the first model.
    if not models or set(models) != set(client_weight_multipliers):
        raise ValueError("Models and weights must identify the same nonempty client set")
    weights = list(client_weight_multipliers.values())
    if any(not isinstance(w, (int, float)) or not 0 <= w <= 1 for w in weights) or abs(sum(weights)-1) > 1e-6:
        raise ValueError("Client weights must be nonnegative and sum to one")
    aggregated_model = copy.deepcopy(next(iter(models.values()))) if aggregated_model is None else copy.deepcopy(aggregated_model)

    for key in aggregated_model.state_dict().keys():
        aggregated_model_parameter = aggregated_model.state_dict(keep_vars=True)[key]

        temp = torch.zeros_like(aggregated_model_parameter, dtype=torch.float32)

        for client_id in models:
            temp += client_weight_multipliers[client_id] * models[client_id].state_dict()[key]

        aggregated_model_parameter.data.copy_(temp)

    return aggregated_model


def get_client_weight_multipliers__nr_of_elements(nr_of_elements_per_client: dict):
    """
    Returns a dictionary of weight multipliers for each client for use in the FedAvg algorithm.
    Each multiplier is the proportion of the client's data relative to the total number of data points across all clients.
    """
    nr_of_total_elements = sum(nr_of_elements_per_client.values())

    return {client_id: nr_of_elements_per_client[client_id] / nr_of_total_elements for client_id in nr_of_elements_per_client.keys()}


def synchronize_models_in_place(aggregated_model, client_models: dict):
    for key in aggregated_model.state_dict().keys():
        for client_model in client_models.values():
            client_model.state_dict()[key].data.copy_(aggregated_model.state_dict()[key])


def get_communication_size_for_model_in_bytes(model, only_count_trainable_parameters=False):
    """Represents the communication size of communicating the given model's parameters. Note that this is not equal to the memory size of loading the model in-memory due to overhead, especially not when using CUDA."""
    return sum(p.numel() * p.element_size() for p in model.parameters()
               if not only_count_trainable_parameters or p.requires_grad)
