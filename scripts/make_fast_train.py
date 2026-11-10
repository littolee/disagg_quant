"""Generate the single-Spark throughput variant, preserving upstream source."""
from pathlib import Path
r=Path('/home/litto/work/disagg_quant')
s=(r/'disaggregated-quantization-main/qad/training/qad.py').read_text()
def replace(a,b):
    global s
    assert s.count(a)==1,(a,s.count(a))
    s=s.replace(a,b)
replace('    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})','    # Spark: retain activations instead of recomputing the forward pass.')
replace('            labels_data = torch.stack(lbl_list).to(device)  # [mbs, T]', '''            labels_data = torch.stack(lbl_list).to(device)  # [mbs, T]
            end = max(int(m.sum()) for m in msk_list)
            input_ids = input_ids[:, :end]
            labels_data = labels_data[:, :end]''')
replace('                t_lm_w = teacher_text.head.weight','''                valid = labels != -100
                s_hidden_q = s_hidden_q[valid].contiguous()
                t_hidden_f = t_hidden_f[valid].contiguous()
                labels = labels[valid].contiguous()
                t_lm_w = teacher_text.head.weight''')
(r/'scripts/qad_fast_generated.py').write_text(s)
entry=(r/'scripts/train_single.py').read_text().replace('from training import qad','import qad_fast_generated as qad')
(r/'scripts/train_fast.py').write_text(entry)
