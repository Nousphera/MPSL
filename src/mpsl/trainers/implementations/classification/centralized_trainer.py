import torch
from tqdm import tqdm

from mpsl.trainers.implementations.experiment_results import ExperimentResults
from mpsl.trainers.implementations.experiment_trainer import ExperimentTrainer

loss_fn = torch.nn.CrossEntropyLoss()


class CentralizedTrainer(ExperimentTrainer):

    def _perform_epoch(self, experiment_results: ExperimentResults, model, device, dataloader, epoch_nr, optimizer):
        is_in_test_mode = optimizer is None
        model.eval() if is_in_test_mode else model.train()

        nr_of_batches, data_iter = len(dataloader), iter(dataloader)
        total_loss, acc, total_examples = 0, 0, 0

        for batch_nr in tqdm(range(nr_of_batches)):
            if not is_in_test_mode:
                optimizer.zero_grad()

            X, y = next(data_iter)
            # Intentionally only sending labels to the device here. Depending on whether input is multimodal and on the modalities themselves, input has to be handled differently
            # prior to sending (chunks of) it to the device. The model will handle sending the input to the appropriate device, itself.
            y = y.to(device)

            predictions = model(X)
            loss = loss_fn(predictions, y)

            total_loss += loss.item() * len(y)
            total_examples += len(y)

            if not is_in_test_mode:
                loss.backward()
                optimizer.step()

            y_pred_class = torch.argmax(predictions, dim=1)
            acc += (y_pred_class == y).sum().item()

        if total_examples == 0:
            raise ValueError("No samples were processed")
        total_loss /= total_examples
        acc /= total_examples

        experiment_results.add_results(epoch_nr, acc, is_in_test_mode)

        return f'test loss {total_loss} and accuracy {acc}' if is_in_test_mode else f'Finished epoch {epoch_nr} with train loss {total_loss} and accuracy {acc}'

    def train_epoch(self, **kwargs):
        return self._perform_epoch(
            experiment_results=kwargs['experiment_results'],
            model=kwargs['model'],
            device=kwargs['device'],
            dataloader=kwargs['dataloader'],
            epoch_nr=kwargs['epoch_nr'],
            optimizer=kwargs['optimizer']
        )

    def test_epoch(self, **kwargs):
        return self._perform_epoch(
            experiment_results=kwargs['experiment_results'],
            model=kwargs['model'],
            device=kwargs['device'],
            dataloader=kwargs['dataloader'],
            epoch_nr=kwargs['epoch_nr'],
            optimizer=None
        )
