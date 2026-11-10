"""Build a separate training entrypoint using the tested memory optimizations."""
from pathlib import Path
import runpy
r=Path('/home/litto/work/disagg_quant')
runpy.run_path(str(r/'scripts/make_fast_train.py'),run_name='__main__')
s=(r/'scripts/qad_fast_generated.py').read_text()
assert s.count('chunk_size=256,')==2
s=s.replace('chunk_size=256,','chunk_size=1024,')
needle='    head_pair = dual_lm_head_weights(student)'
assert s.count(needle)==1
s=s.replace(needle,'''    from reuse_distillation import ReusableKLLoss
    kl_loss_fn = ReusableKLLoss()
    head_pair = dual_lm_head_weights(student)''')
needle='t_ntp = teacher_ce_fn(t_hidden_f, t_lm_w, t_hidden_f, t_lm_w, true_labels=labels)'
assert s.count(needle)==1
s=s.replace(needle,'t_ntp = kl_loss_fn.last_teacher_ce')
(r/'scripts/qad_memory_generated.py').write_text(s)
entry=(r/'scripts/train_fast.py').read_text().replace('qad_fast_generated','qad_memory_generated')
assert entry.count('qad.main()')==1
entry=entry.replace('qad.main()', 'from cache_weight_views import install\ninstall()\nqad.main()')
(r/'scripts/train_memory.py').write_text(entry)
