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
        'input': prompts(
                    contexts(
                        question(Q) + \
                        information(str(tuples)) + \
                        answer(A)
                    )
                )
    }




# python mk.py > mk.log 2>&1 &

if __name__ == "__main__":

    save_root = "/mnt/petrelfs/tangzecheng/Bilingual_L-CiteEval-Ultra/data/zh/"
    file_name = "qa.jsonl"
    save_dir = f"{save_root}/{file_name}"

    vllmpool = VllmPoolExecutor(
        model_name = "Qwen/Qwen2.5-32B",
        tp_size = 8,
    )

    datas = load_dataset("/mnt/petrelfs/tangzecheng/NLPCC-MH", split = 'train')    

    samples = [mkprompt(datas[i]) for i in range(20)]


    result_list = vllmpool.submit(
        prompts = samples
    )

    save_jsonl(result_list, save_dir)
