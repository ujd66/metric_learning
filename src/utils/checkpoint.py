import torch


def save_checkpoint(model, optimizer, epoch, path, **kwargs):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    state.update(kwargs)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = ckpt.get("epoch", 0)
    return epoch, {k: v for k, v in ckpt.items() if k not in ("model_state_dict", "optimizer_state_dict", "epoch")}
