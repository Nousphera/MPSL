import torch


class WrapperModel(torch.nn.Module):

    def __init__(self, client_model, server_model):
        super().__init__()

        self.client_model = client_model
        self.server_model = server_model

    def forward(self, x):
        x = self.client_model(x)
        x = self.server_model(x)

        return x

    def switch_to_device(self, device):
        self.client_model = self.client_model.switch_to_device(device)
        self.server_model = self.server_model.switch_to_device(device)

        return self
