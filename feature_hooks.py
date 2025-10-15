# feature_hooks.py

from typing import Dict, List, Callable, Optional
import torch
import torch.nn as nn
import re

def _iter_tensors(o):
    """Yield all torch.Tensors recursively from nested (list/tuple/dict)."""
    if isinstance(o, torch.Tensor):
        yield o
    elif isinstance(o, (list, tuple)):
        for it in o:
            yield from _iter_tensors(it)
    elif isinstance(o, dict):
        for it in o.values():
            yield from _iter_tensors(it)

def _pick_tensor(o):
    """
    Recursively search output 'o' and pick a good feature map:
    1) prefer 4D (B,C,H,W), else 2) any tensor. No boolean ops on tensors.
    """
    cand4 = None
    cand_any = None
    for t in _iter_tensors(o):
        if cand_any is None:
            cand_any = t
        if t.ndim == 4 and cand4 is None:
            cand4 = t
    return cand4 if cand4 is not None else cand_any  # may be None if no tensors

class FeatureHooker:
    """
    Forward hooks by module-name. Optionally pass selectors[name]: output->tensor override.
    """
    def __init__(self, model: nn.Module, layer_names: List[str], use_regex: bool=False,
                 selectors: Optional[Dict[str, Callable]] = None):
        self.model = model
        self.layer_names = layer_names
        self.use_regex = use_regex
        self.selectors = selectors or {}
        self.handles = []
        self.buffers: Dict[str, object] = {}
        self._register()

    def _register(self):
        name_to_module = dict(self.model.named_modules())
        for want in self.layer_names:
            matched = []
            if self.use_regex:
                patt = re.compile(want)
                matched = [n for n in name_to_module.keys() if patt.search(n)]
            else:
                matched = [want] if want in name_to_module else []
            if not matched:
                print(f"[Hook] WARN: no module matched '{want}'")
                continue
            for name in matched:
                mod = name_to_module[name]
                h = mod.register_forward_hook(self._mk(name))
                self.handles.append(h)
                print(f"[Hook] attached -> {name}")

    def _mk(self, name: str):
        sel = self.selectors.get(name, None)
        def hook(module, inputs, output):
            try:
                self.buffers[name] = sel(output) if sel is not None else _pick_tensor(output)
            except Exception as e:
                print(f"[Hook] output parsing for {name} failed: {e}")
                self.buffers[name] = _pick_tensor(output)
        return hook

    def clear(self): self.buffers.clear()
    def remove(self):
        for h in self.handles: h.remove()
        self.handles.clear()
