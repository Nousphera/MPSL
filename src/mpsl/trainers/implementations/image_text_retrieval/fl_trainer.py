import torch
from tqdm import tqdm

from mpsl.models import InputModality
from mpsl.trainers.implementations.experiment_results import ExperimentResults
from mpsl.trainers.implementations.experiment_trainer import ExperimentTrainer
from mpsl.trainers.implementations.image_text_retrieval.loss import image_text_retrieval_loss
from mpsl.trainers.implementations.image_text_retrieval.metrics import compute_recall_at_k
from mpsl.utils.fl_utils import get_communication_size_for_model_in_bytes, AGGREGATED_MODEL_NAME
from mpsl.utils.mpsl_utils import bytes_to_megabytes, get_client_name

loss_fn = image_text_retrieval_loss
metric = compute_recall_at_k


class FLTrainer(ExperimentTrainer):

    def __init__(self, nr_of_captions_per_image, test_set_size, batch_size, k_vals=[1, 5, 10], move_dist_matrix_to_cpu=False):
        super()

        self.nr_of_captions_per_image = nr_of_captions_per_image
        self.test_set_size = test_set_size
        self.batch_size = batch_size
        self.k_vals = k_vals
        self.move_dist_matrix_to_cpu = move_dist_matrix_to_cpu

        self.cpu_device = torch.device('cpu')

    def _train_epoch(self, desired_device, experiment_results: ExperimentResults, client_data_iterators: dict, client_models: dict, epoch_nr, client_optimizers: dict, client_schedulers: dict):
        [model.train() for model in client_models.values()]

        client_ids = client_models.keys()
        total_loss = 0
        client_specific_metric_dict = {_client_id: 0 for _client_id in client_ids}
        client_specific_loss = {_client_id: 0 for _client_id in client_ids}
        nr_of_elements_per_client = {_client_id: 0 for _client_id in client_ids}

        max_nr_of_batches_in_epoch = 0

        for client_id in tqdm(client_ids):
            client_models[client_id] = client_models[client_id].switch_to_device(desired_device)

            nr_of_batches = len(client_data_iterators[client_id])
            max_nr_of_batches_in_epoch += nr_of_batches

            for batch_nr in tqdm(range(nr_of_batches)):
                try:
                    batch = next(client_data_iterators[client_id])
                    image, text = batch

                    nr_of_elements_per_client[client_id] += len(image)
                except StopIteration as exception:
                    # When this error is thrown, the iterator has no remaining elements, which might occur when some clients have more batches than others.
                    print(exception)
                    continue

                client_optimizers[client_id].zero_grad()

                predictions = client_models[client_id]({
                    InputModality.IMAGE: image,
                    InputModality.TEXT: text
                })
                image, text = predictions[InputModality.IMAGE], predictions[InputModality.TEXT]

                loss = None
                nr_of_correct_predictions = 0

                for text_idx in range(self.nr_of_captions_per_image):
                    itc_loss, _nr_of_correct_predictions = loss_fn(image, text[:, text_idx, :])
                    loss = itc_loss if loss is None else loss + itc_loss

                    nr_of_correct_predictions += _nr_of_correct_predictions

                total_loss += loss.item()
                client_specific_loss[client_id] += loss.item()

                loss.backward()
                client_optimizers[client_id].step()

                client_specific_metric_dict[client_id] += nr_of_correct_predictions

            client_specific_loss[client_id] /= nr_of_batches

            if client_schedulers is not None:
                client_schedulers[client_id].step()

            # Move the model back to the cpu to avoid OutOfMemory issues on the GPU.
            client_models[client_id] = client_models[client_id].switch_to_device(self.cpu_device)

        nr_of_clients = len(client_models.keys())
        total_loss /= max_nr_of_batches_in_epoch

        experiment_results.add_results(epoch_nr, total_loss, False)

        return_string = f"\nFinished epoch {epoch_nr} with total train loss {total_loss}"

        total_client_outgoing_communication_size = 0
        total_client_incoming_communication_size = 0

        total_server_outgoing_communication_size = 0
        total_server_incoming_communication_size = 0

        for client_id in client_models.keys():
            client_specific_formatted_acc = client_specific_metric_dict[client_id]

            # = Communication tracking =
            client_outgoing_comms = get_communication_size_for_model_in_bytes(client_models[client_id], only_count_trainable_parameters=True)
            # The incoming communications are the parameters of the aggregated model, which will have the same shape of the client's model itself.
            client_incoming_comms = client_outgoing_comms

            total_server_incoming_communication_size += client_outgoing_comms
            total_server_outgoing_communication_size += client_outgoing_comms

            total_client_outgoing_communication_size += client_outgoing_comms
            total_client_incoming_communication_size += client_incoming_comms

            return_string += f'\nClient-specific nr_of_correct_predictions for {get_client_name(client_id)}: {client_specific_formatted_acc} with loss: {client_specific_loss[client_id]} and communication overhead: incoming {bytes_to_megabytes(client_incoming_comms)} MB & outgoing {bytes_to_megabytes(client_outgoing_comms)} MB'

        # = Communication tracking =
        avg_incoming_comms_overhead_in_mb = bytes_to_megabytes(total_client_incoming_communication_size / nr_of_clients)
        avg_outgoing_comms_overhead_in_mb = bytes_to_megabytes(total_client_outgoing_communication_size / nr_of_clients)
        print(f'Average client communication overhead: incoming {avg_incoming_comms_overhead_in_mb} MB & outgoing {avg_outgoing_comms_overhead_in_mb} MB')
        experiment_results.set_client_communication_overhead(avg_incoming_comms_overhead_in_mb, avg_outgoing_comms_overhead_in_mb)

        return return_string, nr_of_elements_per_client

    def _test_epoch(self, desired_device, experiment_results: ExperimentResults, dataloader, client_models: dict, epoch_nr):
        print('Initializing test data iterators')
        client_data_iterators = {_client_id: iter(dataloader) for _client_id in client_models.keys()}

        results_as_string = ''

        for client_id in client_models.keys():
            with torch.no_grad():
                client_models[client_id] = client_models[client_id].switch_to_device(desired_device)
                client_name = AGGREGATED_MODEL_NAME if str(client_id) == AGGREGATED_MODEL_NAME else get_client_name(client_id)

                t2i_recall, i2t_recall = metric(desired_device, client_models[client_id], client_data_iterators[client_id], self.k_vals, self.batch_size,
                                                self.nr_of_captions_per_image, self.test_set_size,
                                                self.move_dist_matrix_to_cpu)

                should_add_experiment_results = client_name == AGGREGATED_MODEL_NAME

                if should_add_experiment_results:
                    t2i_recall_as_numerics_list = []
                    i2t_recall_as_numerics_list = []

                # Move the model back to the cpu to avoid OutOfMemory issues on the GPU.
                client_models[client_id] = client_models[client_id].switch_to_device(self.cpu_device)

                results_as_string += f'\nEpoch {epoch_nr} test results: for {client_name}\n'

                results_as_string += 'Text-to-image Recall@K\n'
                for k, x in zip(self.k_vals, t2i_recall):
                    value = f'{100 * x:.2f}'
                    results_as_string += f" R@{k}: {value}%\n"

                    if should_add_experiment_results:
                        t2i_recall_as_numerics_list.append(float(value))

                results_as_string += 'Image-to-text Recall@K\n'
                for k, x in zip(self.k_vals, i2t_recall):
                    value = f'{100 * x:.2f}'
                    results_as_string += f" R@{k}: {value}%\n"

                    if should_add_experiment_results:
                        i2t_recall_as_numerics_list.append(float(value))

        if should_add_experiment_results:
            experiment_results.add_results(epoch_nr, [t2i_recall_as_numerics_list, i2t_recall_as_numerics_list], True)

        return results_as_string

    def train_epoch(self, **kwargs):
        return self._train_epoch(
            desired_device=kwargs['device'],
            experiment_results=kwargs['experiment_results'],
            client_data_iterators=kwargs['client_data_iterators'],
            client_models=kwargs['client_models'],
            epoch_nr=kwargs['epoch_nr'],
            client_optimizers=kwargs['client_optimizers'],
            client_schedulers=kwargs['client_schedulers']
        )

    def test_epoch(self, **kwargs):
        return self._test_epoch(
            desired_device=kwargs['device'],
            experiment_results=kwargs['experiment_results'],
            dataloader=kwargs['dataloader'],
            client_models=kwargs['client_models'],
            epoch_nr=kwargs['epoch_nr']
        )
