import hashlib, requests
from pathlib import Path
root=Path('/home/litto/work/disagg_quant/data/TIGER-Lab__MMLU-Pro'); root.mkdir(parents=True,exist_ok=True)
items={'test-00000-of-00001.parquet':'0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8',
       'validation-00000-of-00001.parquet':'139423c23722e480c807ac4a191409a710cfce4eba744c1d641cf88e730e2078'}
for name,digest in items.items():
 p=root/name
 if p.exists() and hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()==digest: continue
 with requests.get('https://hf-mirror.com/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/'+name,stream=True,timeout=(30,120)) as response:
  response.raise_for_status()
  with open(str(p)+'.part','wb') as f:
   for chunk in response.iter_content(1024*1024): f.write(chunk)
 tmp=Path(str(p)+'.part')
 assert hashlib.file_digest(open(tmp,'rb'),'sha256').hexdigest()==digest
 tmp.replace(p); print('verified',name,flush=True)
