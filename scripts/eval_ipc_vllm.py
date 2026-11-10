import os,sys,json,argparse
from pathlib import Path
import datasets
root=Path("/home/litto/work/disagg_quant")
load=datasets.load_dataset
def local(path,*args,**kw):
    if path=="openai/gsm8k":
        b=root/"data/openai__gsm8k/main"
        return load("parquet",data_files={"train":str(b/"train-00000-of-00001.parquet"),"test":str(b/"test-00000-of-00001.parquet")})
    if path=="HuggingFaceH4/MATH-500": return load("json",data_files={"test":str(root/"data/HuggingFaceH4__MATH-500/test.jsonl")})
    if path=="TIGER-Lab/MMLU-Pro":
        b=root/"data/TIGER-Lab__MMLU-Pro"
        return load("parquet",data_files={"test":str(b/"test-00000-of-00001.parquet"),"validation":str(b/"validation-00000-of-00001.parquet")})
    return load(path,*args,**kw)
datasets.load_dataset=local
from lm_eval.models.openai_completions import LocalCompletionsAPI
from lm_eval import evaluator
parser=argparse.ArgumentParser()
parser.add_argument("--task",required=True)
parser.add_argument("--limit",type=int)
parser.add_argument("--think",action="store_true")
parser.add_argument("--concurrency",type=int,default=8)
args=parser.parse_args()
model="/home/litto/work/models/Qwen/Qwen3-0.6B"
lm=LocalCompletionsAPI(model="pdshared",base_url="http://127.0.0.1:18041/v1/completions",tokenizer=model,tokenizer_backend="huggingface",tokenized_requests=True,num_concurrent=args.concurrency,batch_size=1,max_length=8192,max_gen_toks=4096,timeout=1200)
if not args.think:
    orig=lm.tokenizer.apply_chat_template
    def patched(*a,**kw):
        kw["enable_thinking"]=False
        return orig(*a,**kw)
    lm.tokenizer.apply_chat_template=patched
    probe=lm.tokenizer.apply_chat_template([{"role":"user","content":"hi"}],tokenize=False,add_generation_prompt=True)
    if "<think>\n\n</think>" not in probe: raise RuntimeError("No-think template validation failed")
shot=4 if args.task=="minerva_math500" else 5
result=evaluator.simple_evaluate(model=lm,tasks=[args.task],num_fewshot=shot,apply_chat_template=True,gen_kwargs="max_gen_toks=4096",limit=args.limit,log_samples=True)
mode="think" if args.think else "nothink"
out=root/"results/ipc-vllm-eval"/mode/args.task
out.mkdir(parents=True,exist_ok=True)
suffix="full" if args.limit is None else f"limit{args.limit}"
outpath=out/(suffix+".json")
outpath.write_text(json.dumps(result,default=str,ensure_ascii=False))
print("RESULTS",result["results"],"SAVED",outpath,flush=True)