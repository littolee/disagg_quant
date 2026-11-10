import requests,json,time
base={"model":"pdshared","prompt":"This is a simple factual question. Please consider the question carefully and respond precisely. Many European cities are important capitals. France is a European country whose capital is a very famous city. The question is: what is the capital city of France? Answer:","temperature":0,"max_tokens":16,"stream":False}
p=dict(base,max_tokens=1,kv_transfer_params={"do_remote_decode":True,"do_remote_prefill":False,"remote_engine_id":None,"remote_block_ids":None,"remote_host":None,"remote_port":None})
a=requests.post("http://127.0.0.1:18140/v1/completions",json=p,timeout=120); a.raise_for_status(); j=a.json()
k=j.get("kv_transfer_params")
if not k or not k.get("remote_block_ids") or not k.get("do_remote_prefill"): raise RuntimeError(f"missing transferable KV: {k}")
print("P_KV",json.dumps(k),flush=True)
b=requests.post("http://127.0.0.1:18240/v1/completions",json=dict(base,kv_transfer_params=k),timeout=180)
print("D_STATUS",b.status_code,"D_RESPONSE",b.text,flush=True); b.raise_for_status()
if not b.json().get("choices"): raise RuntimeError("No choices")