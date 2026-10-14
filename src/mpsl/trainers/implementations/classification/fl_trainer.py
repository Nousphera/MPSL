import torch
from tqdm import tqdm

from mpsl.trainers.implementations.experiment_results import ExperimentResults
from mpsl.trainers.implementations.experiment_trainer import ExperimentTrainer
from mpsl.utils.fl_utils import get_communication_size_for_model_in_bytes, AGGREGATED_MODEL_NAME
from mpsl.utils.mpsl_utils import bytes_to_megabytes, get_client_name

loss_fn = torch.nn.CrossEntropyLoss()


class FLTrainer(ExperimentTrainer):

    def __init__(self):
        super()

        self.cpu_device = torch.device('cpu')

    def _perform_epoch(
            self,
            desired_device,
            experiment_results: ExperimentResults,
            client_data_iterators: dict,
            client_models: dict,
            epoch_nr,
            client_optimizers: dict = None,
            client_schedulers: dict = None,
            dataloader_length: int = -1
    ):
        is_in_test_mode = client_optimizers is None
        [model.eval() if is_in_test_mode else model.train() for model in client_models.values()]

        client_ids = client_models.keys()
        total_loss, total_avg_acc, max_nr_of_batches_in_epoch = 0, 0, 0
        client_specific_acc, client_specific_loss, nr_of_elements_per_client = {_client_id: 0 for _client_id in client_ids}, {_client_id: 0 for _client_id in client_ids}, {_client_id: 0 for _client_id in client_ids}

        for client_id in tqdm(client_ids):
            client_models[client_id] = client_models[client_id].switch_to_device(desired_device)
            nr_of_batches = len(client_data_iterators[client_id]) if dataloader_length == -1 else dataloader_length
            max_nr_of_batches_in_epoch += nr_of_batches

            for batch_nr in tqdm(range(nr_of_batches)):
                try:
                    X, y = next(client_data_iterators[client_id])
                    # Intentionally only sending labels to the device here. Depending on whether input is multimodal and the modalities themselves, input has to be handled differently
                    # prior to sending (chunks of) it to the device.
                    y = y.to(desired_device)

                    nr_of_elements_per_client[client_id] += len(y)
                except StopIteration as exception:
                    # When this error is thrown, the iterator has no remaining elements, which might occur when some clients have more batches than others.
                    print(exception)
                    continue

                if not is_in_test_mode:
                    client_optimizers[client_id].zero_grad()

                predictions = client_models[client_id](X)
                loss = loss_fn(predictions, y)

                total_loss += loss.item() * len(y)
                client_specific_loss[client_id] += loss.item() * len(y)

                if not is_in_test_mode:
                    loss.backward()
                    client_optimizers[client_id].step()

                y_pred_class = torch.argmax(predictions, dim=1)
                current_batch_acc = (y_pred_class == y).sum().item()
                total_avg_acc += current_batch_acc

                client_specific_acc[client_id] += current_batch_acc

            if nr_of_elements_per_client[client_id] == 0:
                raise ValueError(f"No samples for client {client_id}")
            client_specific_loss[client_id] /= nr_of_elements_per_client[client_id]
            client_specific_acc[client_id] /= nr_of_elements_per_client[client_id]

            if client_schedulers is not None:
                client_schedulers[client_id].step()

            # Move the model back to the cpu to avoid OutOfMemory issues on the GPU.
            client_models[client_id] = client_models[client_id].switch_to_device(self.cpu_device)

        nr_of_clients = len(client_models.keys())
        total_examples = sum(nr_of_elements_per_client.values())
        if total_examples == 0:
            raise ValueError("No samples were processed")
        total_loss /= total_examples
        total_avg_acc /= total_examples

        experiment_results.add_results(epoch_nr, total_avg_acc, is_in_test_mode)

        return_string = f"\nFinished epoch {epoch_nr} with total {'test' if is_in_test_mode else 'train'} loss {total_loss} and total avg accuracy {total_avg_acc}"

        total_client_outgoing_communication_size, total_client_incoming_communication_size = 0, 0
        total_server_outgoing_communication_size, total_server_incoming_communication_size = 0, 0

        for client_id in client_models.keys():
            # = Communication tracking =
            client_outgoing_comms = get_communication_size_for_model_in_bytes(client_models[client_id], only_count_trainable_parameters=True)
            # The incoming communications are the parameters of the aggregated model, which will have the same shape of the client's model itself.
            client_incoming_comms = client_outgoing_comms

            total_server_incoming_communication_size += client_outgoing_comms
            total_server_outgoing_communication_size += client_outgoing_comms

            total_client_outgoing_communication_size += client_outgoing_comms
            total_client_incoming_communication_size += client_incoming_comms

            client_name = AGGREGATED_MODEL_NAME if str(client_id) == AGGREGATED_MODEL_NAME else get_client_name(client_id)
            return_string += f'\nClient-specific accuracy for {client_name}: {client_specific_acc[client_id]} with loss: {client_specific_loss[client_id]} and communication overhead: incoming {bytes_to_megabytes(client_incoming_comms)} MB & outgoing {bytes_to_megabytes(client_outgoing_comms)} MB'

        # = Communication tracking =
        avg_incoming_comms_overhead_in_mb = bytes_to_megabytes(total_client_incoming_communication_size / nr_of_clients)
        avg_outgoing_comms_overhead_in_mb = bytes_to_megabytes(total_client_outgoing_communication_size / nr_of_clients)
        print(f'Average client communication overhead: incoming {avg_incoming_comms_overhead_in_mb} MB & outgoing {avg_outgoing_comms_overhead_in_mb} MB')
        experiment_results.set_client_communication_overhead(avg_incoming_comms_overhead_in_mb, avg_outgoing_comms_overhead_in_mb)

        return return_string, nr_of_elements_per_client

    def train_epoch(self, **kwargs):
        return self._perform_epoch(
            desired_device=kwargs['device'],
            experiment_results=kwargs['experiment_results'],
            client_data_iterators=kwargs['client_data_iterators'],
            client_models=kwargs['client_models'],
            epoch_nr=kwargs['epoch_nr'],
            client_optimizers=kwargs['client_optimizers'],
            client_schedulers=kwargs['client_schedulers']
        )

    def test_epoch(self, **kwargs):
        print('Initializing test data iterators')

        with torch.no_grad():
            return_string, _ = self._perform_epoch(
                desired_device=kwargs['device'],
                experiment_results=kwargs['experiment_results'],
                client_data_iterators={_client_id: iter(kwargs['dataloader']) for _client_id in kwargs['client_models'].keys()},
                client_models=kwargs['client_models'],
                epoch_nr=kwargs['epoch_nr'],
                client_optimizers=None,
                client_schedulers=None,
                dataloader_length=len(kwargs['dataloader'])
            )
            return return_string
