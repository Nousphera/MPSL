import torch
from tqdm import tqdm

from mpsl.models.modalities import InputModality
from mpsl.models.misc.wrapper_model import WrapperModel
from mpsl.trainers.implementations.experiment_trainer import ExperimentTrainer
from mpsl.trainers.implementations.image_text_retrieval.loss import image_text_retrieval_loss
from mpsl.trainers.implementations.image_text_retrieval.metrics import compute_recall_at_k
from mpsl.trainers.implementations.experiment_results import ExperimentResults
from mpsl.utils.mpsl_utils import get_communication_size, compute_aggregated_loss, bytes_to_megabytes, get_client_name

loss_fn = image_text_retrieval_loss
metric = compute_recall_at_k


class MPSLTrainer(ExperimentTrainer):

    def __init__(self, nr_of_captions_per_image, test_set_size, batch_size, k_vals=[1, 5, 10], move_dist_matrix_to_cpu=False):
        super()

        self.nr_of_captions_per_image = nr_of_captions_per_image
        self.test_set_size = test_set_size
        self.batch_size = batch_size
        self.k_vals = k_vals
        self.move_dist_matrix_to_cpu = move_dist_matrix_to_cpu
        self.modalities = [InputModality.IMAGE, InputModality.TEXT]

    def _train_epoch(self,
                     experiment_results: ExperimentResults,
                     server_model,
                     server_optimizer,
                     client_data_iterators: dict,
                     client_models: dict,
                     client_model_requires_any_grad,
                     client_optimizers: dict,
                     client_schedulers: dict,
                     max_nr_of_batches_in_epoch,
                     epoch_nr,
                     global_args
                     ):
        [model.train() for model in client_models.values()]
        server_model.train()

        client_ids = range(global_args.nr_of_clients)
        total_server_loss, total_nr_correct = 0, 0

        client_specific_metric_dict, nr_of_elements_per_client = {_client_id: 0 for _client_id in client_ids}, {_client_id: 0 for _client_id in client_ids}

        # = Communication tracking =
        client_outgoing_communication_sizes, client_incoming_communication_sizes = {_client_id: 0 for _client_id in client_ids}, {_client_id: 0 for _client_id in client_ids}
        server_outgoing_communication_size, server_incoming_communication_size = 0, 0

        for batch_nr in tqdm(range(max_nr_of_batches_in_epoch)):
            server_optimizer.zero_grad()

            image_intermediate_activations_per_client, text_intermediate_activations_per_client = dict(), dict()
            client_loss_fn_tuples = []

            # We combine the activations of all client mini-batches into a single large batch to speed up training, as it only requires a single forward pass on the server.
            intermediate_activations_entire_batch_combined_image, intermediate_activations_entire_batch_combined_text = None, None

            # The indices that correspond to each client's mini-batch, given that the full batch is the concatenation of all mini-batches together
            mini_batch_indices = dict()

            for client_id in range(global_args.nr_of_clients):
                try:
                    batch = next(client_data_iterators[client_id])
                    image, text = batch

                    nr_of_elements_per_client[client_id] += len(image)
                except StopIteration as exception:
                    # When this error is thrown, the iterator has no remaining elements, which might occur when some clients have more batches than others.
                    print(exception)
                    continue

                # FP on client-side model
                client_optimizers[client_id].zero_grad()
                intermediate_activations = client_models[client_id]({
                    InputModality.IMAGE: image,
                    InputModality.TEXT: text
                })
                image_intermediate_activations_per_client[client_id] = intermediate_activations[InputModality.IMAGE]
                text_intermediate_activations_per_client[client_id] = intermediate_activations[InputModality.TEXT]

                # = Communication tracking =
                comms_size = get_communication_size(image_intermediate_activations_per_client[client_id]) + get_communication_size(text_intermediate_activations_per_client[client_id])
                client_outgoing_communication_sizes[client_id] += comms_size
                server_incoming_communication_size += comms_size

                if intermediate_activations_entire_batch_combined_image is None or intermediate_activations_entire_batch_combined_text is None:
                    intermediate_activations_entire_batch_combined_image = image_intermediate_activations_per_client[client_id]
                    intermediate_activations_entire_batch_combined_text = text_intermediate_activations_per_client[client_id]

                    mini_batch_indices[client_id] = (0, len(image_intermediate_activations_per_client[client_id]))
                else:
                    client_begin_index = len(intermediate_activations_entire_batch_combined_image)
                    mini_batch_indices[client_id] = (client_begin_index, client_begin_index + len(image))

                    intermediate_activations_entire_batch_combined_image = torch.cat((intermediate_activations_entire_batch_combined_image, image_intermediate_activations_per_client[client_id]))
                    intermediate_activations_entire_batch_combined_text = torch.cat((intermediate_activations_entire_batch_combined_text, text_intermediate_activations_per_client[client_id]))

            # Single full batch FP on server-side model for each modality
            final_activations_image = intermediate_activations_entire_batch_combined_image.detach().clone().requires_grad_(True)
            final_activations_text = intermediate_activations_entire_batch_combined_text.detach().clone().requires_grad_(True)
            final_predictions = server_model({
                InputModality.IMAGE: final_activations_image,
                InputModality.TEXT: final_activations_text
            })
            image, text = final_predictions[InputModality.IMAGE], final_predictions[InputModality.TEXT]

            for client_id in range(global_args.nr_of_clients):
                if client_id not in mini_batch_indices:
                    continue

                # Loss computation on client-side
                client_mini_batch_indices = mini_batch_indices[client_id]
                image_for_client = image[client_mini_batch_indices[0]:client_mini_batch_indices[1]]
                text_for_client = text[client_mini_batch_indices[0]:client_mini_batch_indices[1]]

                mini_batch_length = len(image_for_client)

                client_loss_fn = None
                client_nr_correct = 0

                for text_idx in range(self.nr_of_captions_per_image):
                    itc_loss, _client_nr_correct = loss_fn(image_for_client, text_for_client[:, text_idx, :])
                    client_loss_fn = itc_loss if client_loss_fn is None else client_loss_fn + itc_loss

                    client_nr_correct += _client_nr_correct

                client_loss_fn_tuples.append((client_loss_fn, mini_batch_length))

                # = Communication tracking =
                comms_size = get_communication_size(image_for_client) + get_communication_size(text_for_client)
                client_incoming_communication_sizes[client_id] += comms_size
                server_outgoing_communication_size += comms_size

                comms_size = get_communication_size(client_loss_fn)
                client_outgoing_communication_sizes[client_id] += comms_size
                server_incoming_communication_size += comms_size

                # Metrics
                total_nr_correct += client_nr_correct

                client_specific_metric = client_specific_metric_dict[client_id]
                client_specific_metric_dict[client_id] = client_specific_metric + client_nr_correct

            # BP on server-side
            agg_loss = compute_aggregated_loss(client_loss_fn_tuples, len(intermediate_activations_entire_batch_combined_image))

            total_server_loss += agg_loss.item()

            agg_loss.backward()

            server_optimizer.step()

            # We should only compute gradients and perform client-side BP if client-side models actually require gradients.
            if client_model_requires_any_grad:
                for final_activations, intermediate_client_activations, is_last_modality in [(final_activations_image, image_intermediate_activations_per_client, False), (final_activations_text, text_intermediate_activations_per_client, True)]:
                    # Server-side 'sending' gradients & client-side performing BP
                    for client_id in range(global_args.nr_of_clients):
                        if client_id in mini_batch_indices:
                            (client_begin_index, client_end_index) = mini_batch_indices[client_id]
                            cut_layer_grads = final_activations.grad[client_begin_index:client_end_index].clone()

                            intermediate_client_activations[client_id].backward(cut_layer_grads)

                            if is_last_modality:
                                client_optimizers[client_id].step()

                            # = Communication tracking =
                            comms_size = get_communication_size(cut_layer_grads)
                            client_incoming_communication_sizes[client_id] += comms_size
                            server_outgoing_communication_size += comms_size

        total_server_loss /= max_nr_of_batches_in_epoch

        experiment_results.add_results(epoch_nr, total_server_loss, False)

        print(f'Finished training epoch with server communication overhead: incoming {bytes_to_megabytes(server_incoming_communication_size)} MB & outgoing {bytes_to_megabytes(server_outgoing_communication_size)} MB')

        total_client_outgoing_communication_size, total_client_incoming_communication_size = 0, 0

        for client_id in range(global_args.nr_of_clients):
            # = Communication tracking =
            client_outgoing_comms = client_outgoing_communication_sizes[client_id]
            client_incoming_comms = client_incoming_communication_sizes[client_id]
            total_client_outgoing_communication_size += client_outgoing_comms
            total_client_incoming_communication_size += client_incoming_comms

            print(f'Client-specific train itc avg_nr_correct for {get_client_name(client_id)}: {client_specific_metric_dict[client_id]} with communication overhead: incoming {bytes_to_megabytes(client_incoming_comms)} MB & outgoing {bytes_to_megabytes(client_outgoing_comms)} MB')

            if client_model_requires_any_grad:
                client_schedulers[client_id].step()

        # = Communication tracking =
        avg_incoming_comms_overhead_in_mb = bytes_to_megabytes(total_client_incoming_communication_size / global_args.nr_of_clients)
        avg_outgoing_comms_overhead_in_mb = bytes_to_megabytes(total_client_outgoing_communication_size / global_args.nr_of_clients)
        print(f'Average client communication overhead: incoming {avg_incoming_comms_overhead_in_mb} MB & outgoing {avg_outgoing_comms_overhead_in_mb} MB')

        experiment_results.set_client_communication_overhead(avg_incoming_comms_overhead_in_mb, avg_outgoing_comms_overhead_in_mb)

        return f'Finished epoch {epoch_nr} with train loss {total_server_loss} and itc total_nr_correct: {total_nr_correct}', nr_of_elements_per_client

    def _test_epoch(self, device, experiment_results: ExperimentResults, server_model, client_model, dataloader, epoch_nr):
        client_model.eval()
        server_model.eval()

        with torch.no_grad():
            wrapper_model = WrapperModel(client_model, server_model)

            t2i_recall, i2t_recall = metric(device, wrapper_model, iter(dataloader), self.k_vals, self.batch_size, self.nr_of_captions_per_image, self.test_set_size, self.move_dist_matrix_to_cpu)

            t2i_recall_as_numerics_list = []
            i2t_recall_as_numerics_list = []

            results_as_string = f'Epoch {epoch_nr} test results:\n'

            results_as_string += 'Text-to-image Recall@K\n'
            for k, x in zip(self.k_vals, t2i_recall):
                value = f'{100 * x:.2f}'
                results_as_string += f" R@{k}: {value}%\n"

                t2i_recall_as_numerics_list.append(float(value))

            results_as_string += 'Image-to-text Recall@K\n'
            for k, x in zip(self.k_vals, i2t_recall):
                value = f'{100 * x:.2f}'
                results_as_string += f" R@{k}: {value}%\n"

                i2t_recall_as_numerics_list.append(float(value))

            experiment_results.add_results(epoch_nr, [t2i_recall_as_numerics_list, i2t_recall_as_numerics_list], True)

            return results_as_string

    def train_epoch(self, **kwargs):
        return self._train_epoch(
            experiment_results=kwargs['experiment_results'],
            server_model=kwargs['server_model'],
            server_optimizer=kwargs['server_optimizer'],
            client_data_iterators=kwargs['client_data_iterators'],
            client_models=kwargs['client_models'],
            client_model_requires_any_grad=kwargs['client_model_requires_any_grad'],
            client_optimizers=kwargs['client_optimizers'],
            client_schedulers=kwargs['client_schedulers'],
            max_nr_of_batches_in_epoch=kwargs['max_nr_of_batches_in_epoch'],
            epoch_nr=kwargs['epoch_nr'],
            global_args=kwargs['global_args']
        )

    def test_epoch(self, **kwargs):
        return self._test_epoch(
            device=kwargs['device'],
            experiment_results=kwargs['experiment_results'],
            server_model=kwargs['server_model'],
            client_model=kwargs['client_model'],
            dataloader=kwargs['dataloader'],
            epoch_nr=kwargs['epoch_nr']
        )
