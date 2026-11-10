import time,subprocess,urllib.request,json,re
from pathlib import Path
root=Path('/home/litto/work/disagg_quant'); out=root/'results/latency-breakdown/metrics-run'; out.mkdir(parents=True,exist_ok=True)
ports={'p':18140,'d':18260,'bf16':18360}
names=['time_to_first_token_seconds','e2e_request_latency_seconds','request_queue_time_seconds','request_inference_time_seconds','request_prefill_time_seconds','request_decode_time_seconds']
def snap():
 result={}
 for tag,port in ports.items():
  text=urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics').read().decode(); a={}
  for line in text.splitlines():
   for name in names:
    for suffix in ('sum','count'):
     if line.startswith(f'vllm:{name}_{suffix}'+'{'):
      a[name+'_'+suffix]=float(line.rsplit(' ',1)[1])
  result[tag]=a
 return result
print('settling',flush=True); time.sleep(12)
data={}
for mode,port in [('pd',18061),('bf16',18360)]:
 before=snap(); print('start',mode,flush=True)
 subprocess.run([str(root/'.venv-eval/bin/python'),'-u',str(root/'scripts/trace_latency_metrics_client.py'),mode,str(port)],check=True)
 time.sleep(12); after=snap()
 data[mode]={tag:{key:val-before[tag].get(key,0) for key,val in vals.items()} for tag,vals in after.items()}
 (out/'metric-deltas.json').write_text(json.dumps(data,indent=2)); print('done',mode,flush=True)
print(json.dumps(data,indent=2),flush=True)
