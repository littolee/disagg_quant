import json,re,statistics
from pathlib import Path
root=Path('/home/litto/work/disagg_quant'); base=root/'results/latency-breakdown'
clients=json.loads((base/'pd-client.json').read_text())
proxies={}
for line in (root/'logs/ipc-profile-proxy-wall.log').read_text().splitlines():
 if line.startswith('PROFILE '):
  p=json.loads(line[8:]); proxies[p['id']]=p
rx=re.compile(r'IPC_PROFILE request=(cmpl-[a-f0-9]+).*?start=(\d+) map_start=(\d+) map_end=(\d+) copy_start=(\d+) submit_end=(\d+) wait_end=(\d+) blocks=(\d+)')
copies={}
for line in (root/'logs/ipc-profile-decode-wall.log').read_text().splitlines():
 m=rx.search(line)
 if m: copies[m[1]]=list(map(int,m.groups()[1:]))
records=[]
for c in clients:
 p=proxies.get(c['id']); v=copies.get(c['id'])
 if not p or not v: continue
 start,ms,me,cs,se,we,blocks=v
 stages={'client_to_proxy':p['proxy_start']-c['start'],'proxy_before_p':p['p_send']-p['proxy_start'],'p_http_including_engine':p['p_done']-p['p_send'],'proxy_between_p_d':p['d_send']-p['p_done'],'d_before_ipc':start-p['d_send'],'ipc_setup':cs-start,'ipc_submit_cpu':se-cs,'ipc_wait':we-se,'d_after_ipc':p['d_done']-we,'proxy_after_d':p['proxy_end']-p['d_done'],'proxy_to_client':c['end']-p['proxy_end']}
 if any(z<0 for z in stages.values()): raise ValueError((c['id'],stages))
 records.append({'id':c['id'],'blocks':blocks,'prompt_tokens':c['prompt_tokens'],'stages_ms':{k:v/1e6 for k,v in stages.items()},'total_ms':(c['end']-c['start'])/1e6})
assert len(records)==len(clients),(len(records),len(clients))
keys=list(records[0]['stages_ms'])
summary={k:{'mean':statistics.mean(r['stages_ms'][k] for r in records),'p50':statistics.median(r['stages_ms'][k] for r in records)} for k in keys}
summary['total_ms']={'mean':statistics.mean(r['total_ms'] for r in records),'p50':statistics.median(r['total_ms'] for r in records)}
obj={'n':len(records),'units':'milliseconds','summary':summary,'records':records}
(base/'joined-trace.json').write_text(json.dumps(obj,indent=2))
print(json.dumps({'n':len(records),'summary':summary},indent=2))
