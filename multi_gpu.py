"""Layer-wise model placement without splitting the contrastive batch."""

import os

import torch


def resolve_devices(*, require_cuda=False):
    """Default to one GPU; use all visible GPUs when explicitly configured.

    CUDA interprets CUDA_VISIBLE_DEVICES, including UUIDs and device ordering.
    Use its logical device count rather than parsing physical IDs ourselves.
    """
    if not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError("Evaluation requires an available CUDA GPU")
        return [torch.device("cpu")]
    count = torch.cuda.device_count() if "CUDA_VISIBLE_DEVICES" in os.environ else 1
    torch.cuda.set_device(0)
    return [torch.device("cuda", i) for i in range(count)]


def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _place_layer(layer, device, output_device):
    layer.to(device)

    def before_forward(module, args, kwargs):
        return _move(args, device), _move(kwargs, device)

    def after_forward(module, args, output):
        return _move(output, output_device)

    layer.register_forward_pre_hook(before_forward, with_kwargs=True)
    layer.register_forward_hook(after_forward)


def place_model(model, devices):
    """Call once, after loading weights and before creating the optimizer.

    Keep the encoder orchestration on the primary device, including LayerDrop
    decisions. Each off-device layer returns to it, so skipped layers also work.
    Tensor.to preserves autograd edges; module/parameter names stay unchanged.
    """
    if getattr(model, "_model_parallel_placed", False):
        raise ValueError("Model placement has already been configured")
    primary = devices[0]
    if len(devices) == 1:
        return model.to(primary)
    layers = model.ssl.encoder.layers
    if len(devices) > len(layers):
        raise ValueError("Cannot use more GPUs than encoder layers")

    # Place each layer directly; never materialize the full model on GPU 0.
    assignments = {
        id(layer): devices[index * len(devices) // len(layers)]
        for index, layer in enumerate(layers)
    }

    def place(module):
        if id(module) in assignments:
            destination = assignments[id(module)]
            if destination == primary:
                module.to(primary)
            else:
                _place_layer(module, destination, primary)
            return
        # Move only this module's own parameters/buffers, preserving children.
        module._apply(lambda tensor: tensor.to(primary), recurse=False)
        for child in module.children():
            place(child)

    place(model)
    model._model_parallel_placed = True
    return model
