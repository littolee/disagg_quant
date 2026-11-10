"""Opt-in BF16 STE view cached until quantized weights are refreshed."""
from quantizers.blocked import BlockScaledLinear

def install():
    import torch
    original = BlockScaledLinear._differentiable_weight
    original_update = BlockScaledLinear.post_update
    def view(self):
        cached = getattr(self, '_spark_bf16_view', None)
        if cached is None:
            cached = original(self).to(torch.bfloat16)
            self._spark_bf16_view = cached
        return cached
    def update(self, step, total_steps):
        self._spark_bf16_view = None
        return original_update(self, step, total_steps)
    BlockScaledLinear._differentiable_weight = view
    BlockScaledLinear.post_update = update
