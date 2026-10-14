from tqdm import tqdm

from mpsl.models import InputModality
from mpsl.trainers.implementations.experiment_trainer import ExperimentTrainer
from mpsl.trainers.implementations.image_text_retrieval.loss import image_text_retrieval_loss
from mpsl.trainers.implementations.image_text_retrieval.metrics import compute_recall_at_k
from mpsl.trainers.implementations.experiment_results import ExperimentResults

loss_fn = image_text_retrieval_loss
metric = compute_recall_at_k


class CentralizedTrainer(ExperimentTrainer):

    def __init__(self, nr_of_captions_per_image, test_set_size, batch_size, k_vals=[1, 5, 10], move_dist_matrix_to_cpu=False):
        super()

        self.nr_of_captions_per_image = nr_of_captions_per_image
        self.test_set_size = test_set_size
        self.batch_size = batch_size
        self.k_vals = k_vals
        self.move_dist_matrix_to_cpu = move_dist_matrix_to_cpu

    def _train_epoch(self, experiment_results: ExperimentResults, model, dataloader, epoch_nr, optimizer):
        model.train()

        nr_of_batches, data_iter = len(dataloader), iter(dataloader)

        total_loss = 0

        for batch_nr in tqdm(range(nr_of_batches)):
            optimizer.zero_grad()

            batch = next(data_iter)
            image, text = batch

            predictions = model({
                InputModality.IMAGE: image,
                InputModality.TEXT: text
            })
            image, text = predictions[InputModality.IMAGE], predictions[InputModality.TEXT]

            loss = None

            for text_idx in range(self.nr_of_captions_per_image):
                itc_loss, _ = loss_fn(image, text[:, text_idx, :])
                loss = itc_loss if loss is None else loss + itc_loss

            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        total_loss /= nr_of_batches

        experiment_results.add_results(epoch_nr, total_loss, False)

        return f'Finished epoch {epoch_nr} with train loss {total_loss}'

    def _test_epoch(self, experiment_results: ExperimentResults, model, device, dataloader, epoch_nr):
        model.eval()

        t2i_recall, i2t_recall = metric(device, model, iter(dataloader), self.k_vals, self.batch_size, self.nr_of_captions_per_image, self.test_set_size, self.move_dist_matrix_to_cpu)

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
            model=kwargs['model'],
            dataloader=kwargs['dataloader'],
            epoch_nr=kwargs['epoch_nr'],
            optimizer=kwargs['optimizer']
        )

    def test_epoch(self, **kwargs):
        return self._test_epoch(
            experiment_results=kwargs['experiment_results'],
            model=kwargs['model'],
            device=kwargs['device'],
            dataloader=kwargs['dataloader'],
            epoch_nr=kwargs['epoch_nr']
        )
