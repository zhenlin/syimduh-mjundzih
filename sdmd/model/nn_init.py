import torch
import torch.nn as nn


def stretched_eye(in_features: int, out_features: int) -> torch.Tensor:
    a = torch.zeros((out_features, in_features))

    if min(in_features, out_features) == 0:
        return a

    n = max(in_features, out_features)

    for k in range(n):
        i = int(out_features * k / n)
        j = int(in_features * k / n)

        a[i, j] = 1
    
    if in_features < out_features:
        a = a / a.sum(0).sqrt()[None, :]
    elif in_features > out_features:
        a = a / a.sum(1).sqrt()[:, None]

    return a

def stretched_eye_pm(in_features: int, out_features: int) -> torch.Tensor:
    a = torch.empty((out_features, in_features))

    a[0::2, :] = stretched_eye(in_features, (out_features + 1) // 2)
    a[1::2, :] = -stretched_eye(in_features, out_features // 2)

    return a


def Linear(*args, initialization: str = 'stretched_eye', initialization_scale: float = 1.0, **kwargs) -> nn.Linear:
    layer = nn.Linear(*args, **kwargs)

    if initialization == 'stretched_eye':
        layer.weight.data = stretched_eye(layer.in_features, layer.out_features) * initialization_scale

        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif initialization == 'stretched_eye_pm':
        layer.weight.data = stretched_eye_pm(layer.in_features, layer.out_features) * initialization_scale

        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif initialization == 'eye':
        layer.weight.data = torch.eye(layer.out_features, layer.in_features) * initialization_scale

        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif initialization == 'zero':
        nn.init.zeros_(layer.weight)

        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    else:
        raise ValueError(f'{initialization} is not a valid value for initialization')
    
    return layer


def Conv2d(*args, initialization: str = 'dirac', initialization_scale: float = 1.0, **kwargs) -> nn.Conv2d:
    layer = nn.Conv2d(*args, **kwargs)

    if initialization == 'dirac':
        nn.init.dirac_(layer.weight, layer.groups)
        layer.weight.data *= initialization_scale

        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    else:
        raise ValueError(f'{initialization} is not a valid value for initialization')

    return layer


def ConvTranspose2d(*args, initialization: str = 'dirac', initialization_scale: float = 1.0, **kwargs) -> nn.Conv2d:
    layer = nn.ConvTranspose2d(*args, **kwargs)

    if initialization == 'dirac':
        nn.init.dirac_(layer.weight, layer.groups)
        layer.weight.data *= initialization_scale

        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    else:
        raise ValueError(f'{initialization} is not a valid value for initialization')

    return layer