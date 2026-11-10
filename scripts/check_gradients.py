import json, torch, sys
from pathlib import Path
r=Path('/home/litto/work/disagg_quant/results')
prefix=sys.argv[1] if len(sys.argv)>1 else 'grads'
a=torch.load(r/(prefix+'-0.pt'),weights_only=False,map_location='cpu')
b=torch.load(r/(prefix+'-1.pt'),weights_only=False,map_location='cpu')
out={'loss_base':a['loss'],'loss_compact':b['loss'],'loss_abs_diff':abs(a['loss']-b['loss']), 'tensors':len(a['grads'])}
assert a['grads'].keys()==b['grads'].keys()
ref=diff=0.; maxabs=0.
for n,x in a['grads'].items():
    y=b['grads'][n]; d=(x-y).float()
    ref+=float(x.float().square().sum()); diff+=float(d.square().sum()); maxabs=max(maxabs,float(d.abs().max()))
out.update(gradient_relative_l2=(diff/max(ref,1e-30))**.5,gradient_max_abs=maxabs)
(r/(prefix+'_validation.json')).write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
