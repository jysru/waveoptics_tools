import torch


def pearson(x, y, inverse: bool = False, squared: bool = False):
    x, y = torch.abs(x), torch.abs(y)
    if squared:
        x, y = torch.square(x), torch.square(y)
    mx, my = torch.mean(x), torch.mean(y)
    sx, sy = torch.std(x), torch.std(y)
    p = torch.sum((x - mx) * (y - my)) / x.numel()
    p = p / (sx * sy)
    return 1 - p if inverse else p


def quality(x, y, inverse: bool = False, squared: bool = True):
    q = torch.abs(torch.sum(x * torch.conj(y)) / torch.sum(torch.abs(x) * torch.abs(y)))
    q = torch.square(q) if squared else q
    return 1 - q if inverse else q


def energy_in_target(y, target, inverse: bool = False):
    p = torch.sum(torch.square(torch.abs(y * target))) / torch.sum(torch.square(torch.abs(y)))
    return 1 - p if inverse else p


def power_overlap_integral(y, target, inverse: bool = False):
    numer = torch.square(torch.abs(torch.sum(y * torch.conj(target))))
    denom = torch.sum(torch.square(torch.abs(y))) * torch.sum(torch.square(torch.abs(target)))
    over = numer / denom
    return 1 - over if inverse else over


def power_overlap_integral_moduli(y, target, inverse: bool = False):
    numer = torch.square(torch.abs(torch.sum(torch.abs(y) * torch.abs(target))))
    denom = torch.sum(torch.square(torch.abs(y))) * torch.sum(torch.square(torch.abs(target)))
    over = numer / denom
    return 1 - over if inverse else over
