"""Run upstream vLLM evaluator using checked local GSM8K/MATH datasets."""
from pathlib import Path
import sys,datasets
root=Path('/home/litto/work/disagg_quant')
load=datasets.load_dataset
def local(path,*args,**kwargs):
    if path=='openai/gsm8k':
        b=root/'data/openai__gsm8k/main'
        return load('parquet',data_files={'train':str(b/'train-00000-of-00001.parquet'),'test':str(b/'test-00000-of-00001.parquet')})
    if path=='HuggingFaceH4/MATH-500':
        return load('json',data_files={'test':str(root/'data/HuggingFaceH4__MATH-500/test.jsonl')})
    if path=='TIGER-Lab/MMLU-Pro':
        b=root/'data/TIGER-Lab__MMLU-Pro'
        return load('parquet',data_files={'test':str(b/'test-00000-of-00001.parquet'),'validation':str(b/'validation-00000-of-00001.parquet')})
    return load(path,*args,**kwargs)
datasets.load_dataset=local
sys.path.insert(0,str(root/'disaggregated-quantization-main/qad/eval'))
import eval_vllm
eval_vllm.main()
