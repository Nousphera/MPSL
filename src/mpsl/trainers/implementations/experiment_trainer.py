from abc import abstractmethod


class ExperimentTrainer:

    @abstractmethod
    def train_epoch(self, **kwargs):
        pass

    @abstractmethod
    def test_epoch(self, **kwargs):
        pass
