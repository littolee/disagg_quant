"""Single-GPU fake-quant evaluator adapted from upstream; not native vLLM."""
import os,sys,hashlib
from pathlib import Path
root=Path('/home/litto/work/disagg_quant')
import datasets
_load_dataset=datasets.load_dataset
def local_dataset(path,*args,**kwargs):
    if path=='openai/gsm8k':
        base=root/'data/openai__gsm8k/main'
        return _load_dataset('parquet',data_files={'train':str(base/'train-00000-of-00001.parquet'),
                                                   'test':str(base/'test-00000-of-00001.parquet')})
    if path=='HuggingFaceH4/MATH-500':
        return _load_dataset('json',data_files={'test':str(root/'data/HuggingFaceH4__MATH-500/test.jsonl')})
    if path=='TIGER-Lab/MMLU-Pro':
        base=root/'data/TIGER-Lab__MMLU-Pro'
        return _load_dataset('parquet',data_files={'test':str(base/'test-00000-of-00001.parquet'),
                                                   'validation':str(base/'validation-00000-of-00001.parquet')})
    return _load_dataset(path,*args,**kwargs)
datasets.load_dataset=local_dataset
sys.path.insert(0,str(root/'disaggregated-quantization-main/qad/eval'))
import eval_transformers as ev
_load=ev.load_model
def load_sdpa(model,dtype,attn_implementation=None):
    return _load(model,dtype,attn_implementation='sdpa')
ev.load_model=load_sdpa
from lm_eval import evaluator
_evaluate=evaluator.simple_evaluate
def evaluate_with_chat(*args,**kwargs):
    kwargs['apply_chat_template']=True
    cache_dir=root/'cache/lm-eval'
    cache_dir.mkdir(parents=True,exist_ok=True)
    key=hashlib.sha256('\0'.join(sys.argv[1:]).encode()).hexdigest()[:24]
    kwargs['use_cache']=str(cache_dir/key)
    return _evaluate(*args,**kwargs)
evaluator.simple_evaluate=evaluate_with_chat
if '--no-think' in sys.argv:
    original=ev.AutoTokenizer.from_pretrained
    def load_tokenizer(*args,**kwargs):
        tok=original(*args,**kwargs)
        apply=tok.apply_chat_template
        def render(*a,**k):
            k['enable_thinking']=False
            return apply(*a,**k)
        tok.apply_chat_template=render
        return tok
    ev.AutoTokenizer.from_pretrained=load_tokenizer
ev.main()
