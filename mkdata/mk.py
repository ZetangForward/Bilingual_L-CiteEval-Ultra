from datasets import load_dataset

from deploy import *
# import sys
# sys.path.append("..")
from mkdata_prompt import *



def mkprompt(info: dict):
    tuples = info['path']
    for t in tuples:
        for i in range(len(t)):
            t[i] = t[i].split(" ||| ")[0]

    Q = info['q']
    A = tuples[-1][-1]

    return {
        'question': Q,
        'answer'  : A,
        'facts'   : tuples,
        'input'   : prompts(tuples)
    }




# python mk.py > mk.log 2>&1 &

if __name__ == "__main__":

    # save_root = "/mnt/petrelfs/tangzecheng/Bilingual_L-CiteEval-Ultra/data/zh/"
    save_root = "../data/zh"
    file_name = "qa.jsonl"
    save_dir = f"{save_root}/{file_name}"

    vllmpool = VllmPoolExecutor(
        model_name = "/data/hf_models/Qwen2-57B-A14B-Instruct",
        # model_name = "/data/hf_models/Meta-Llama-3.1-8B-Instruct",
        # model_name = "/data/hf_models/Meta-Llama-3.1-70B-Instruct",
        tp_size = 4,
    )

    datas = load_dataset("../data/NLPCC-MH", split = 'train')    

    samples = [mkprompt(datas[i]) for i in range(20)]


    result_list = vllmpool.submit(
        prompts = samples
    )

    save_jsonl(result_list, save_dir)
