from modelzipper.tutils import *
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("/data/hf_models/Meta-Llama-3.1-8B-Instruct")
counter_fact = auto_read_data("/data/zky/iclr2024/dataset/niah/niah_processed_counterfact_llama31_8b_64000.json")
wiki_64k = auto_read_data("/data/zky/iclr2024/dataset/2WikiMultihopQA/data_ids/64000/2wiki_64000_final.json")
hothop_64k = auto_read_data("/data/zky/iclr2024/dataset/2WikiMultihopQA/data_ids/64000/2wiki_64000_final.json")

counter_fact = Dataset.from_list(counter_fact)

def add_k(item):
    all_str = ''.join(item['docs'])
    length = len(tokenizer(all_str)['input_ids'])

    item['length'] = length
    item['hardness'] = None
    item.pop('ground_truth')
    return item

def add_k_2(item):
    all_str = ''.join([x['text'] for x in item['docs']])
    length = len(tokenizer(all_str)['input_ids'])

    item['length'] = length
    item['hardness'] = None
    item.pop('gold_docs')
    return item

counter_fact = counter_fact.map(add_k, num_proc=32)

wiki_64k_dataset = Dataset.from_list(wiki_64k)
wiki_64k_dataset = wiki_64k_dataset.map(add_k_2, num_proc=32)

hothop_64k_dataset = Dataset.from_list(hothop_64k)
hothop_64k_dataset = hothop_64k_dataset.map(add_k_2, num_proc=32)

datasets = ["narrativeqa", "natural_questions", "counting_stars", "niah"]

origin_dict = {}
for dataset in datasets:  ### Load L-CiteEval
    origin_dict[dataset] = load_dataset('/root/.cache/huggingface/hub/datasets--Jonaszky123--L-CiteEval/snapshots/c79c928529593f478e6573c969cf73d22f0cf0f9', f"L-CiteEval-Data_{dataset}", trust_remote_code=True)

new_dict = {
    'multihop_qa': Dataset.from_list([x for x in hothop_64k_dataset] + [x for x in wiki_64k_dataset]).shuffle(seed=42).select(range(120)),
    'single_qa': Dataset.from_list([x for x in origin_dict['narrativeqa']['test']] + [x for x in origin_dict['natural_questions']['test']]).shuffle(seed=42).select(range(120)),
    'counterfact': Dataset.from_list([x for x in counter_fact]).shuffle(seed=42).select(range(120)),
    'counting_stars': origin_dict['counting_stars']['test'].shuffle(seed=42).select(range(120)),
    'niah': origin_dict['niah']['test'].shuffle(seed=42).select(range(120))
}

new_dataset = Dataset.from_dict(new_dict)

new_dataset = DatasetDict({"en_citeeval": new_dataset})

new_dataset.push_to_hub("ZetangForward/ZH_CiteEval", private=True, token="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")