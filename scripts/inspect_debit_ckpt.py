#!/usr/bin/env python
"""Read the released DEBiT checkpoint's tensors without installing Habitat.

The .pth is a habitat_baselines PPO checkpoint whose `config` entry pickles
habitat / omegaconf / hydra classes we do not have, so a plain torch.load dies
with ModuleNotFoundError before reaching a single tensor. Every class from a
module we lack is resolved to an inert placeholder instead: the tensors
unpickle normally and the config unpickles into stand-ins nobody reads.

usage (as a script): inspect_debit_ckpt.py CKPT.pth   -- prints the param tree
usage (as a module):  from inspect_debit_ckpt import load_tensors
"""
import pickle
import sys

import torch


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        try:
            self.__dict__.update(state)
        except Exception:
            pass

    def __reduce__(self):
        return (_Stub, ())


class Unpickler(pickle.Unpickler):
    MISSING = ("habitat", "habitat_baselines", "omegaconf", "hydra")

    def find_class(self, module, name):
        if any(module == m or module.startswith(m + ".") for m in self.MISSING):
            return _Stub
        return super().find_class(module, name)


def load(f, **kw):
    """torch.load(pickle_module=...) entry point."""
    kw.pop("encoding", None)
    return Unpickler(f, **kw).load()


def load_tensors(path):
    """{param name: tensor} from the checkpoint's state_dict."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False,
                      pickle_module=sys.modules[__name__])
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    return {k: v for k, v in sd.items() if torch.is_tensor(v)}


if __name__ == "__main__":
    t = load_tensors(sys.argv[1])
    print(f"tensors: {len(t)}  params: {sum(v.numel() for v in t.values())/1e6:.1f} M")
    groups = {}
    for k, v in t.items():
        g = ".".join(k.split(".")[:3])
        groups[g] = groups.get(g, 0) + v.numel()
    for g, n in sorted(groups.items(), key=lambda x: -x[1])[:15]:
        print(f"   {g:58s} {n/1e6:8.2f} M")
