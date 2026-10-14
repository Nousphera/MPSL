import torch
from tqdm import tqdm

from mpsl.models.modalities import InputModality
from mpsl.trainers.implementations.experiment_results import ExperimentResults
from mpsl.trainers.implementations.experiment_trainer import ExperimentTrainer
from mpsl.utils.mpsl_utils import get_communication_size, compute_aggregated_loss, bytes_to_megabytes, get_client_name

from mpsl.utils.split_transport import TensorBoundary

loss_fn = torch.nn.CrossEntropyLoss()


class MPSLTrainer(ExperimentTrainer):

    def __init__(self, fusion_type, modalities: list[InputModality], transport=None):
        super()

        self.is_using_late_fusion = 'late' in str(fusion_type).lower()
        self.modalities = modalities
        self.transfer = TensorBoundary(transport)

    def _early_fusion_epoch(self, device, server_model, server_optimizer, client_data_iterators: dict, client_models: dict, client_model_requires_any_grad, client_optimizers: dict, max_nr_of_batches_in_epoch, global_args):
        client_ids = range(global_args.nr_of_clients)
        total_server_loss, acc = 0, 0
        client_specific_acc_tuples, nr_of_elements_per_client = {_client_id: (0, 0) for _client_id in client_ids}, {_client_id: 0 for _client_id in client_ids}

        # = Communication tracking =
        client_outgoing_communication_sizes, client_incoming_communication_sizes = {_client_id: 0 for _client_id in client_ids}, {_client_id: 0 for _client_id in client_ids}
        server_outgoing_communication_size, server_incoming_communication_size = 0, 0

        for batch_nr in tqdm(range(max_nr_of_batches_in_epoch)):
            server_optimizer.zero_grad()

            # We combine the activations of all client mini-batches into a single large batch to speed up training, as it only requires a single forward pass on the server.
            intermediate_activations_entire_batch_combined = None
            intermediate_activations_per_client, client_loss_fn_tuples = dict(), []

            # The indices that correspond to each client's mini-batch, given that the full batch is the concatenation of all mini-batches together
            mini_batch_indices, y_per_client = dict(), dict()

            for client_id in range(global_args.nr_of_clients):
                try:
                    X, y = next(client_data_iterators[client_id])

                    # Intentionally only sending labels to the device here. Depending on whether input is multimodal and the modalities themselves, input has to be handled differently
                    # prior to sending (chunks of) it to the device.
                    y = y.to(device)
                    y_per_client[client_id] = y

                    nr_of_elements_per_client[client_id] += len(y)
                except StopIteration as exception:
                    # When this error is thrown, the iterator has no remaining elements, which might occur when some clients have more batches than others.
                    print(exception)
                    continue

                # FP on client-side model
                client_optimizers[client_id].zero_grad()
                intermediate_activations = client_models[client_id](X)
                intermediate_activations_per_client[client_id] = intermediate_activations

                # = Communication tracking =
                comms_size = get_communication_size(intermediate_activations)
                client_outgoing_communication_sizes[client_id] += comms_size
                server_incoming_communication_size += comms_size

                # Combine all intermediate activations to allow for a single FP
                if intermediate_activations_entire_batch_combined is None:
                    intermediate_activations_entire_batch_combined = self.transfer(intermediate_activations)

                    mini_batch_indices[client_id] = (0, len(intermediate_activations))
                else:
                    client_begin_index = len(intermediate_activations_entire_batch_combined)
                    mini_batch_indices[client_id] = (client_begin_index, client_begin_index + len(y))

                    intermediate_activations_entire_batch_combined = torch.cat((intermediate_activations_entire_batch_combined, self.transfer(intermediate_activations)))

            # Single full batch FP on server-side model
            final_activations = intermediate_activations_entire_batch_combined.detach().clone().requires_grad_(True)
            predictions = server_model(final_activations)
            prediction_gradients = torch.zeros_like(predictions)

            for client_id in range(global_args.nr_of_clients):
                if client_id not in mini_batch_indices:
                    continue

                # Loss computation on client-side
                client_mini_batch_indices = mini_batch_indices[client_id]
                preds_for_client = self.transfer(predictions[client_mini_batch_indices[0]:client_mini_batch_indices[1]]).requires_grad_(True)
                y_for_client = y_per_client[client_id]
                mini_batch_length = len(y_for_client)

                client_loss_fn = loss_fn(preds_for_client, y_for_client)
                # The loss graph stays on the client. Send its weighted logit gradient.
                logit_gradient, = torch.autograd.grad(
                    client_loss_fn * (mini_batch_length / len(predictions)), preds_for_client
                )
                logit_gradient = self.transfer(logit_gradient)
                prediction_gradients[client_mini_batch_indices[0]:client_mini_batch_indices[1]] = logit_gradient
                loss_payload = self.transfer(client_loss_fn)
                client_loss_fn_tuples.append((loss_payload, mini_batch_length))

                # = Communication tracking =
                comms_size = get_communication_size(preds_for_client)
                client_incoming_communication_sizes[client_id] += comms_size
                server_outgoing_communication_size += comms_size

                comms_size = get_communication_size(logit_gradient) + get_communication_size(loss_payload)
                client_outgoing_communication_sizes[client_id] += comms_size
                server_incoming_communication_size += comms_size

                # Metrics
                y_pred_class = torch.argmax(preds_for_client, dim=1)
                current_batch_acc = (y_pred_class == y_for_client).sum().item()
                acc += current_batch_acc

                client_specific_acc, client_specific_total_batch_size = client_specific_acc_tuples[client_id]
                client_specific_acc_tuples[client_id] = (client_specific_acc + current_batch_acc, client_specific_total_batch_size + mini_batch_length)

            # BP on server-side
            agg_loss = compute_aggregated_loss(client_loss_fn_tuples, len(intermediate_activations_entire_batch_combined))

            total_server_loss += agg_loss.item() * sum(size for _, size in client_loss_fn_tuples)

            predictions.backward(prediction_gradients)

            server_optimizer.step()

            # We should only compute gradients and perform client-side BP if client-side models actually require gradients.
            if client_model_requires_any_grad:
                # Server-side 'sending' gradients & client-side performing BP
                for client_id in range(global_args.nr_of_clients):
                    if client_id in mini_batch_indices:
                        (client_begin_index, client_end_index) = mini_batch_indices[client_id]
                        cut_layer_grads = self.transfer(final_activations.grad[client_begin_index:client_end_index])

                        intermediate_activations_per_client[client_id].backward(cut_layer_grads)
                        client_optimizers[client_id].step()

                        # = Communication tracking =
                        comms_size = get_communication_size(cut_layer_grads)
                        client_incoming_communication_sizes[client_id] += comms_size
                        server_outgoing_communication_size += comms_size

        return total_server_loss, \
            acc, \
            server_incoming_communication_size, \
            server_outgoing_communication_size, \
            client_incoming_communication_sizes, \
            client_outgoing_communication_sizes, \
            client_specific_acc_tuples, \
            nr_of_elements_per_client

    def _late_fusion_epoch(self, device, server_model, server_optimizer, client_data_iterators: dict, client_models: dict, client_model_requires_any_grad, client_optimizers: dict, max_nr_of_batches_in_epoch, global_args):
        client_ids = range(global_args.nr_of_clients)
        total_server_loss, acc = 0, 0
        client_specific_acc_tuples, nr_of_elements_per_client = {_client_id: (0, 0) for _client_id in client_ids}, {_client_id: 0 for _client_id in client_ids}

        # = Communication tracking =
        client_outgoing_communication_sizes, client_incoming_communication_sizes = {_client_id: 0 for _client_id in client_ids}, {_client_id: 0 for _client_id in client_ids}
        server_outgoing_communication_size, server_incoming_communication_size = 0, 0

        for batch_nr in tqdm(range(max_nr_of_batches_in_epoch)):
            # A dict that will contain a dict of all client models' activations for all given modalities.
            intermediate_activations_dict_for_modality = {modality: dict() for modality in self.modalities}
            # We combine the activations of all client mini-batches into a single large batch to speed up training, as it only requires a single forward pass on the server.
            intermediate_activations_entire_batch_combined_for_modality = {modality: None for modality in self.modalities}

            client_loss_fn_tuples = []
            server_optimizer.zero_grad()

            # The indices that correspond to each client's mini-batch, given that the full batch is the concatenation of all mini-batches together
            mini_batch_indices, y_per_client = dict(), dict()

            for client_id in range(global_args.nr_of_clients):
                try:
                    X, y = next(client_data_iterators[client_id])

                    # Intentionally only sending labels to the device here. Depending on whether input is multimodal and the modalities themselves, input has to be handled differently
                    # prior to sending (chunks of) it to the device.
                    y = y.to(device)
                    y_per_client[client_id] = y

                    nr_of_elements_per_client[client_id] += len(y)
                except StopIteration as exception:
                    # When this error is thrown, the iterator has no remaining elements, which might occur when some clients have more batches than others.
                    print(exception)
                    continue

                # FP on client-side model
                client_optimizers[client_id].zero_grad()
                intermediate_activations = client_models[client_id](X)

                for modality in self.modalities:
                    intermediate_activations_dict_for_modality[modality][client_id] = intermediate_activations[modality]

                    # = Communication tracking =
                    comms_size = get_communication_size(intermediate_activations[modality])
                    client_outgoing_communication_sizes[client_id] += comms_size
                    server_incoming_communication_size += comms_size

                    # Combine all intermediate activations for each modality to allow for a single FP per modality
                    if intermediate_activations_entire_batch_combined_for_modality[modality] is None:
                        intermediate_activations_entire_batch_combined_for_modality[modality] = self.transfer(intermediate_activations[modality])

                        mini_batch_indices[client_id] = (0, len(intermediate_activations[modality]))
                    else:
                        client_begin_index = len(intermediate_activations_entire_batch_combined_for_modality[modality])

                        if client_id not in mini_batch_indices:
                            mini_batch_indices[client_id] = (client_begin_index, client_begin_index + len(y))

                        intermediate_activations_entire_batch_combined_for_modality[modality] = torch.cat((intermediate_activations_entire_batch_combined_for_modality[modality], self.transfer(intermediate_activations[modality])))

            # Single full batch FP on server-side model
            final_activations_dict_for_modality = {modality: intermediate_activations_entire_batch_combined_for_modality[modality].detach().clone().requires_grad_(True) for modality in self.modalities}
            predictions = server_model(final_activations_dict_for_modality)
            prediction_gradients = torch.zeros_like(predictions)

            for client_id in range(global_args.nr_of_clients):
                if client_id not in mini_batch_indices:
                    continue

                # Loss computation on client-side
                client_mini_batch_indices = mini_batch_indices[client_id]
                preds_for_client = self.transfer(predictions[client_mini_batch_indices[0]:client_mini_batch_indices[1]]).requires_grad_(True)
                y_for_client = y_per_client[client_id]
                mini_batch_length = len(y_for_client)

                client_loss_fn = loss_fn(preds_for_client, y_for_client)
                # The loss graph stays on the client. Send its weighted logit gradient.
                logit_gradient, = torch.autograd.grad(
                    client_loss_fn * (mini_batch_length / len(predictions)), preds_for_client
                )
                logit_gradient = self.transfer(logit_gradient)
                prediction_gradients[client_mini_batch_indices[0]:client_mini_batch_indices[1]] = logit_gradient
                loss_payload = self.transfer(client_loss_fn)
                client_loss_fn_tuples.append((loss_payload, mini_batch_length))

                # = Communication tracking =
                comms_size = get_communication_size(preds_for_client)
                client_incoming_communication_sizes[client_id] += comms_size
                server_outgoing_communication_size += comms_size

                comms_size = get_communication_size(logit_gradient) + get_communication_size(loss_payload)
                client_outgoing_communication_sizes[client_id] += comms_size
                server_incoming_communication_size += comms_size

                # Metrics
                y_pred_class = torch.argmax(preds_for_client, dim=1)
                current_batch_acc = (y_pred_class == y_for_client).sum().item()
                acc += current_batch_acc

                client_specific_acc, client_specific_total_batch_size = client_specific_acc_tuples[client_id]
                client_specific_acc_tuples[client_id] = (client_specific_acc + current_batch_acc, client_specific_total_batch_size + mini_batch_length)

            # BP on server-side
            agg_loss = compute_aggregated_loss(client_loss_fn_tuples, len(intermediate_activations_entire_batch_combined_for_modality[self.modalities[0]]))

            total_server_loss += agg_loss.item() * sum(size for _, size in client_loss_fn_tuples)

            predictions.backward(prediction_gradients)

            server_optimizer.step()

            # We should only compute gradients and perform client-side BP if client-side models actually require gradients.
            if client_model_requires_any_grad:
                for idx, modality in enumerate(self.modalities):
                    is_last_modality = idx == len(self.modalities) - 1

                    # Server-side 'sending' gradients & client-side performing BP
                    for client_id in range(global_args.nr_of_clients):
                        if client_id in mini_batch_indices:
                            (client_begin_index, client_end_index) = mini_batch_indices[client_id]
                            final_activations = final_activations_dict_for_modality[modality]
                            cut_layer_grads = self.transfer(final_activations.grad[client_begin_index:client_end_index])

                            intermediate_activations_dict_for_modality[modality][client_id].backward(cut_layer_grads)

                            if is_last_modality:
                                client_optimizers[client_id].step()

                            # = Communication tracking =
                            comms_size = get_communication_size(cut_layer_grads)
                            client_incoming_communication_sizes[client_id] += comms_size
                            server_outgoing_communication_size += comms_size

        return total_server_loss, \
            acc, \
            server_incoming_communication_size, \
            server_outgoing_communication_size, \
            client_incoming_communication_sizes, \
            client_outgoing_communication_sizes, \
            client_specific_acc_tuples, \
            nr_of_elements_per_client

    def _train_epoch(self, device, server_model, server_optimizer, client_data_iterators, client_models, client_model_requires_any_grad, client_optimizers, max_nr_of_batches_in_epoch, client_schedulers, experiment_results: ExperimentResults, epoch_nr, global_args):
        """
        We are using activations combining; We combine all (modality-specific) client-side activations into a single large batch, requiring only a single FP on the server-model. This speeds up training.
        """
        [model.train() for model in client_models.values()]
        server_model.train()

        if self.is_using_late_fusion:
            total_server_loss, \
                acc, \
                server_incoming_communication_size, \
                server_outgoing_communication_size, \
                client_incoming_communication_sizes, \
                client_outgoing_communication_sizes, \
                client_specific_acc_tuples, \
                nr_of_elements_per_client_dict = self._late_fusion_epoch(
                    device,
                    server_model,
                    server_optimizer,
                    client_data_iterators,
                    client_models,
                    client_model_requires_any_grad,
                    client_optimizers,
                    max_nr_of_batches_in_epoch,
                    global_args
                )
        else:
            total_server_loss, \
                acc, \
                server_incoming_communication_size, \
                server_outgoing_communication_size, \
                client_incoming_communication_sizes, \
                client_outgoing_communication_sizes, \
                client_specific_acc_tuples, \
                nr_of_elements_per_client_dict = self._early_fusion_epoch(
                    device,
                    server_model,
                    server_optimizer,
                    client_data_iterators,
                    client_models,
                    client_model_requires_any_grad,
                    client_optimizers,
                    max_nr_of_batches_in_epoch,
                    global_args)

        total_examples = sum(nr_of_elements_per_client_dict.values())
        if total_examples == 0:
            raise ValueError("No training samples were processed")
        total_server_loss /= total_examples
        acc /= total_examples

        experiment_results.add_results(epoch_nr, acc, False)

        print(f'Finished training epoch with server communication overhead: incoming {bytes_to_megabytes(server_incoming_communication_size)} MB & outgoing {bytes_to_megabytes(server_outgoing_communication_size)} MB')

        total_client_outgoing_communication_size, total_client_incoming_communication_size = 0, 0

        for client_id in range(global_args.nr_of_clients):
            client_specific_total_acc, client_specific_total_batch_size = client_specific_acc_tuples[client_id]

            # = Communication tracking =
            client_outgoing_comms, client_incoming_comms = client_outgoing_communication_sizes[client_id], client_incoming_communication_sizes[client_id]
            total_client_outgoing_communication_size += client_outgoing_comms
            total_client_incoming_communication_size += client_incoming_comms

            print(f'Client-specific train accuracy for {get_client_name(client_id)}: {client_specific_total_acc / client_specific_total_batch_size if client_specific_total_batch_size else float("nan")} with communication overhead: incoming {bytes_to_megabytes(client_incoming_comms)} MB & outgoing {bytes_to_megabytes(client_outgoing_comms)} MB')

            if client_model_requires_any_grad:
                client_schedulers[client_id].step()

        # = Communication tracking =
        avg_incoming_comms_overhead_in_mb = bytes_to_megabytes(total_client_incoming_communication_size / global_args.nr_of_clients)
        avg_outgoing_comms_overhead_in_mb = bytes_to_megabytes(total_client_outgoing_communication_size / global_args.nr_of_clients)
        print(f'Average client communication overhead: incoming {avg_incoming_comms_overhead_in_mb} MB & outgoing {avg_outgoing_comms_overhead_in_mb} MB')
        experiment_results.set_client_communication_overhead(avg_incoming_comms_overhead_in_mb, avg_outgoing_comms_overhead_in_mb)

        return f'Finished epoch {epoch_nr} with train loss {total_server_loss} and accuracy {acc}', nr_of_elements_per_client_dict

    def _test_epoch(self, client_model, server_model, dataloader, device, experiment_results: ExperimentResults, epoch_nr):
        client_model.eval()
        server_model.eval()

        total_server_loss, acc = 0, 0

        nr_of_batches, data_iter = len(dataloader), iter(dataloader)
        total_examples = 0

        with torch.no_grad():
            for batch_nr in tqdm(range(nr_of_batches)):
                try:
                    X, y = next(data_iter)
                except StopIteration:
                    # When this error is thrown, the iterator has no remaining elements, which might occur when some clients have more batches than others.
                    continue

                # Intentionally only sending labels to the device here. Depending on whether input is multimodal and the modalities themselves, input has to be handled differently
                # prior to sending (chunks of) it to the device.
                y = y.to(device)

                # FP on client-side and model followed by FP on server-side model
                predictions = server_model(client_model(X))

                # Loss computation on client-side
                client_loss = loss_fn(predictions, y)

                # Metrics
                y_pred_class = torch.argmax(predictions, dim=1)
                current_batch_acc = (y_pred_class == y).sum().item()
                total_examples += len(y)
                acc += current_batch_acc

                # BP on server-side
                batch_size = len(y)
                agg_loss = compute_aggregated_loss([(client_loss, batch_size)], batch_size)

                total_server_loss += agg_loss.item() * len(y)

        if total_examples == 0:
            raise ValueError("No evaluation samples were processed")
        total_server_loss /= total_examples
        acc /= total_examples

        experiment_results.add_results(epoch_nr, acc, True)

        return f'test loss {total_server_loss} and accuracy {acc}'

    def train_epoch(self, **kwargs):
        return self._train_epoch(
            device=kwargs['device'],
            server_model=kwargs['server_model'],
            server_optimizer=kwargs['server_optimizer'],
            client_data_iterators=kwargs['client_data_iterators'],
            client_models=kwargs['client_models'],
            client_model_requires_any_grad=kwargs['client_model_requires_any_grad'],
            client_optimizers=kwargs['client_optimizers'],
            max_nr_of_batches_in_epoch=kwargs['max_nr_of_batches_in_epoch'],
            client_schedulers=kwargs['client_schedulers'],
            experiment_results=kwargs['experiment_results'],
            epoch_nr=kwargs['epoch_nr'],
            global_args=kwargs['global_args']
        )

    def test_epoch(self, **kwargs):
        return self._test_epoch(
            client_model=kwargs['client_model'],
            server_model=kwargs['server_model'],
            dataloader=kwargs['dataloader'],
            device=kwargs['device'],
            experiment_results=kwargs['experiment_results'],
            epoch_nr=kwargs['epoch_nr']
        )
