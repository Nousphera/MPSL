import torch
import torch.nn.functional as F


def log_softmax(x, dim):
    """
    As per https://github.com/OFA-Sys/ONE-PEACE/
    """
    return F.log_softmax(x, dim=dim, dtype=torch.float32)


def adjust_label_smoothed_nll_loss(lprobs, target, epsilon=0.0):
    """
    As per https://github.com/OFA-Sys/ONE-PEACE/
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)

    nll_loss = -lprobs.gather(dim=-1, index=target).squeeze(-1)

    if epsilon != 0:
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True).squeeze(-1)
        eps_i = epsilon / (lprobs.size(-1) - 1)
        loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    else:
        loss = nll_loss

    return loss.mean()


def i2t_loss(image_logits, text_logits, targets=None):
    if targets is None:
        targets = torch.arange(0, image_logits.size(0)).to(image_logits.device)

    sim_i2t = image_logits @ text_logits.T
    log_sim_i2t = log_softmax(sim_i2t, dim=-1).type_as(sim_i2t)

    with torch.no_grad():
        i2t_preds = sim_i2t.argmax(dim=1)
        i2t_ncorrect = (i2t_preds == targets).float().sum()

    return adjust_label_smoothed_nll_loss(log_sim_i2t, targets), i2t_ncorrect


def t2i_loss(image_logits, text_logits, targets=None):
    if targets is None:
        targets = torch.arange(0, image_logits.size(0)).to(image_logits.device)

    sim_t2i = text_logits @ image_logits.T
    log_sim_t2i = log_softmax(sim_t2i, dim=-1).type_as(sim_t2i)

    with torch.no_grad():
        t2i_preds = sim_t2i.argmax(dim=1)
        t2i_ncorrect = (t2i_preds == targets).float().sum()

    return adjust_label_smoothed_nll_loss(log_sim_t2i, targets), t2i_ncorrect


def image_text_retrieval_loss(image_logits, text_logits):
    """
    As per https://github.com/OFA-Sys/ONE-PEACE/

    Final loss is computed as the average of both i2t and t2i losses.
    """
    targets = torch.arange(0, image_logits.size(0))
    targets = targets.to(image_logits.device)

    loss_i2t, i2t_ncorrect = i2t_loss(image_logits, text_logits, targets)
    loss_t2i, t2i_ncorrect = t2i_loss(image_logits, text_logits, targets)

    return ((loss_i2t + loss_t2i) / 2), (i2t_ncorrect + t2i_ncorrect)
