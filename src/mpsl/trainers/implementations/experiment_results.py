from typing import List


class ExperimentResults:
    def __init__(self, arguments=None):
        self.configuration = dict(vars(arguments)) if arguments is not None else None
        self.dataset_provenance = None
        if arguments is not None and arguments.dataset != 'cifar100':
            from pathlib import Path
            from mpsl.available_datasets.preparation import load_manifest, sha256
            manifest = load_manifest(arguments.torch_data_dir, arguments.dataset)
            path = Path(arguments.torch_data_dir) / arguments.dataset / 'manifest.json'
            self.dataset_provenance = {
                'manifest': str(path.resolve()), 'sha256': sha256(path),
                'counts': manifest['counts'], 'class_to_idx': manifest['class_to_idx'],
                'custom_samples': manifest['custom_samples'],
            }
        self.epochs = []
        self.train_metric = []
        self.test_metric = []
        self.incoming_client_communication_overhead_in_mb = -1
        self.outgoing_client_communication_overhead_in_mb = -1

    def add_results(self, epoch_nr, metric, is_in_test_mode):
        if epoch_nr not in self.epochs:
            self.epochs.append(epoch_nr)

        if is_in_test_mode:
            self.test_metric.append(metric)
        else:
            self.train_metric.append(metric)

    def set_client_communication_overhead(self, incoming_in_mb, outgoing_in_mb):
        """
        Sets the incoming and outgoing communication overhead for a client in mb, if not already set.
        Since the overhead is an average over all clients and the dataset length is constant, the values will be identical each epoch. Hence, the values are only set once.
        """
        if self.incoming_client_communication_overhead_in_mb != -1 or self.outgoing_client_communication_overhead_in_mb != -1:
            return

        self.incoming_client_communication_overhead_in_mb = incoming_in_mb
        self.outgoing_client_communication_overhead_in_mb = outgoing_in_mb

    def to_json(self):
        return {
            "configuration": self.configuration,
            "dataset": self.dataset_provenance,
            "epochs": self.epochs,
            "train_metric": self.train_metric,
            "test_metric": self.test_metric,
            "incoming_client_communication_overhead_in_mb": self.incoming_client_communication_overhead_in_mb,
            "outgoing_client_communication_overhead_in_mb": self.outgoing_client_communication_overhead_in_mb
        }
