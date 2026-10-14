"""Detached tensor exchanges for the single-process split-learning simulator."""


class TensorBoundary:
    """Copy tensor values without sharing an autograd graph across endpoints.

    ``transport`` may serialize/deserialize a detached tensor. The default is a
    local clone. Neither path is a network service or a privacy mechanism.
    Communication counters measure logical tensor bytes, not serializer bytes.
    """

    def __init__(self, transport=None):
        self.transport = transport

    def __call__(self, tensor):
        value = tensor.detach()
        received = value.clone() if self.transport is None else self.transport(value)
        return received.detach()
