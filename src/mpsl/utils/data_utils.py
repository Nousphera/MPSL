"""DataLoader initialization shared by distributed training entrypoints."""

from concurrent.futures import ThreadPoolExecutor


def initialize_data_iterators(client_dataloaders):
    """Initialize loaders concurrently and propagate startup errors to the caller."""
    with ThreadPoolExecutor() as executor:
        futures = {
            client_id: executor.submit(iter, loader)
            for client_id, loader in client_dataloaders.items()
        }
        return {client_id: future.result() for client_id, future in futures.items()}
