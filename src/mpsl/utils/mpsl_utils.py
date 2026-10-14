import math


def get_client_name(client_id: int):
    return f'client{client_id}'


def compute_mini_batch_size(server_batch_size, nr_of_clients):
    return math.ceil(server_batch_size / nr_of_clients)


def compute_aggregated_loss(losses_and_mini_batch_sizes, total_batch_size):
    """
    :param losses_and_mini_batch_sizes: An array of tuples of the form (loss_fn, mini-batch size)
    :param total_batch_size: The total real batch size, which is the sum of all mini-batches
    """
    agg_loss = None

    for loss_fn, mini_batch_size in losses_and_mini_batch_sizes:
        if agg_loss is None:
            agg_loss = ((mini_batch_size / total_batch_size) * loss_fn)
        else:
            agg_loss += ((mini_batch_size / total_batch_size) * loss_fn)

    return agg_loss


def client_model_requires_any_grad(client_model):
    for name, param in client_model.named_parameters():
        if param.requires_grad:
            return True
    return False


def get_communication_size(tensor):
    """Tensor payload bytes, excluding allocator, protocol and backing-view overhead."""
    return tensor.numel() * tensor.element_size()


def bytes_to_megabytes(_bytes):
    return _bytes / (1024 * 1024)
