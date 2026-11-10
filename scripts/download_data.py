import concurrent.futures, hashlib, json, pathlib, time, requests
ROOT=pathlib.Path('/home/litto/work/disagg_quant/data')
HOST='https://hf-mirror.com'
def fetch(repo, selected=None):
    tree=requests.get(f'{HOST}/api/datasets/{repo}/tree/main?recursive=true',timeout=40)
    tree.raise_for_status()
    entries=[e for e in tree.json() if e['type']=='file' and e['path'].endswith(('.parquet','.jsonl')) and (selected is None or any(s in e['path'] for s in selected))]
    dest=ROOT/repo.replace('/','__'); dest.mkdir(parents=True,exist_ok=True)
    (dest/'source.json').write_text(json.dumps({'repo':repo,'files':entries},indent=2))
    def one(e):
        p=dest/e['path']; p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists() and p.stat().st_size==e['size']: return
        for attempt in range(5):
            try:
                print('Downloading',repo,e['path'],e['size'],flush=True)
                with requests.get(f"{HOST}/datasets/{repo}/resolve/main/{e['path']}?download=true",stream=True,timeout=(30,120)) as r:
                    r.raise_for_status()
                    with open(str(p)+'.part','wb') as f:
                        for c in r.iter_content(4*1024*1024): f.write(c)
                tmp=pathlib.Path(str(p)+'.part')
                assert tmp.stat().st_size==e['size'],(tmp.stat().st_size,e['size'])
                if 'lfs' in e:
                    assert hashlib.file_digest(open(tmp,'rb'),'sha256').hexdigest()==e['lfs']['oid']
                tmp.replace(p); print('Complete',str(p),flush=True); return
            except Exception as ex:
                print('Retry',attempt,repr(ex),flush=True); time.sleep(3)
        raise RuntimeError(e['path'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool: list(pool.map(one,entries))
    if not entries: raise RuntimeError('No parquet files found: '+repo)
for repo,selected in [('allenai/tulu-3-sft-mixture',None),('HuggingFaceH4/MATH-500',None)]:
    fetch(repo,selected)
