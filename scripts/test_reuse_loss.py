import json,sys,torch
from pathlib import Path
from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss
from reuse_distillation import ReusableKLLoss
torch.manual_seed(42)
r=Path('/home/litto/work/disagg_quant')
device='cuda'
n,h,v=128,1024,151936
x=torch.randn(n,h,device=device,dtype=torch.float32,requires_grad=True)
w=(torch.randn(v,h,device=device,dtype=torch.float32)*.02).requires_grad_()
t=torch.randn(n,h,device=device,dtype=torch.bfloat16)
tw=torch.randn(v,h,device=device,dtype=torch.bfloat16)*.02
labels=torch.randint(v,(n,),device=device)
fn=LigerFusedLinearJSDLoss(weight_hard_loss=0.,weight_soft_loss=1.,beta=0.,compiled=False,chunk_size=1024,return_soft_hard_loss=True)
a=fn(x,w,t,tw,true_labels=labels)
a[0].backward()
gx=x.grad.detach().clone(); gw=w.grad.detach().clone(); x.grad=None; w.grad=None
fn2=ReusableKLLoss(); b=fn2(x,w,t,tw,labels); b[0].backward()
rel=lambda a,b: float((a-b).norm()/a.norm().clamp_min(1e-20))
out={'original_loss':float(a[0].detach()),'candidate_loss':float(b[0].detach()),
     'input_grad_relative_l2':rel(gx,x.grad),'weight_grad_relative_l2':rel(gw,w.grad),
     'student_ce_absolute_difference':float((a[2]-b[2]).abs()),'shape':[n,h,v]}
(r/'results/reuse_loss_validation.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
assert abs(out['original_loss']-out['candidate_loss'])<1e-5
assert out['input_grad_relative_l2']<1e-4 and out['weight_grad_relative_l2']<1e-4
