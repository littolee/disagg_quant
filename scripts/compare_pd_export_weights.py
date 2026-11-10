from pathlib import Path
from safetensors import safe_open
import torch
root=Path("/home/litto/work/disagg_quant/checkpoints-memory/spark-qwen06-memory-nvfp4pdshared-2d557910/weights/step_0002484")
with safe_open(root/"prefill/model.safetensors",framework="pt",device="cpu") as p, safe_open(root/"decode/model.safetensors",framework="pt",device="cpu") as d:
 pk=set(p.keys()); dk=set(d.keys())
 print("only_prefill",len(pk-dk),"only_decode",len(dk-pk))
 groups={}
 for k in sorted(pk&dk):
  name=k.rsplit(".",1)[-1]
  a=p.get_tensor(k); b=d.get_tensor(k)
  same=a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
  counts=groups.setdefault(name,[0,0])
  counts[0]+=1; counts[1]+=int(same)
 for key,value in sorted(groups.items()): print(key,"equal",value[1],"total",value[0])