import torch


def get_optimizer_and_scheduler(model, global_args):
    optimizer = torch.optim.Adam(model.parameters(), lr=global_args.start_lr)

    return optimizer, get_scheduler(optimizer, global_args)


def get_scheduler(optimizer, global_args):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=global_args.scheduler_step_size)
