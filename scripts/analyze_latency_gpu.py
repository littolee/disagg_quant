import re,json,statistics
from pathlib import Path
root=Path('/home/litto/work/disagg_quant'); out=root/'results/latency-breakdown/gpu-run'
ids={v['id'] for v in json.loads((out/'pd-client.json').read_text())}; gpu={}; wall={}
for line in (root/'logs/ipc-profile-decode.log').read_text().splitlines():
 m=re.search(r'IPC_GPU_PROFILE request=(cmpl-[a-f0-9]+).*gpu_ms=([\d.]+)',line)
 if m and m[1] in ids:gpu[m[1]]=float(m[2])
 m=re.search(r'IPC_PROFILE request=(cmpl-[a-f0-9]+).*copy_start=(\d+) submit_end=(\d+) wait_end=(\d+) blocks=(\d+)',line)
 if m and m[1] in ids:wall[m[1]]=list(map(int,m.groups()[1:]))
assert len(gpu)==len(wall)==len(ids),(len(gpu),len(wall),len(ids))
vals=list(gpu.values()); w=[(e-s)/1e6 for s,_,e,_ in wall.values()]; blocks=[x[3] for x in wall.values()]
r={'count':len(vals),'gpu_ms_mean':statistics.mean(vals),'gpu_ms_p50':statistics.median(vals),'gpu_ms_p95':sorted(vals)[int(len(vals)*.95)],'wall_copy_ms_mean':statistics.mean(w),'wall_copy_ms_p50':statistics.median(w),'blocks_mean':statistics.mean(blocks),'blocks_min':min(blocks),'blocks_max':max(blocks)}
(out/'summary.json').write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
