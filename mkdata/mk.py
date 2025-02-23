from datasets import load_dataset

from deploy import *
# import sys
# sys.path.append("..")
from mkdata_prompt import *



def mkprompt(info: dict,index: int, task = 'qa', language = 'zh', dtype :Literal['long', 'short', 'interference']= 'long'):
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
        'input'   : prompts(Q,tuples,
                            task = task, language = language, 
                            dtype = dtype),
        'index' : index
    }




# python mk.py > mk.log 2>&1 &
models = {
    'zh':['internlm/internlm2_5-20b-chat',
          'Qwen/Qwen2.5-32B-Instruct',
          'internlm/internlm2_5-20b-chat']
}

if __name__ == "__main__":

    # save_root = "/mnt/petrelfs/tangzecheng/Bilingual_L-CiteEval-Ultra/data/zh/"
    save_root = "../data/zh"
    

    vllmpool = VllmPoolExecutor(
        # model_name = "/data/hf_models/Qwen2-57B-A14B-Instruct",
        model_name = "Qwen/Qwen2.5-32B-Instruct",
        # model_name = "/data/hf_models/Meta-Llama-3.1-8B-Instruct",
        # model_name = "/data/hf_models/Meta-Llama-3.1-70B-Instruct",
        tp_size = 4,
    )

    datas = load_dataset("/mnt/petrelfs/tangzecheng/NLPCC-MH", split = 'train')    

    qa1 = read_jsonl("/mnt/petrelfs/tangzecheng/Bilingual_L-CiteEval-Ultra/data/zh/source_qa1.jsonl")
    qa2 = datas.filter(lambda x:len(x['path'])==2)
    qa3 = datas.filter(lambda x:len(x['path'])==3)


    for dtype in ['short']:
        for hop, sub_datas in zip(['1', '2', '3'],[qa1, qa2, qa3]):
            if hop=='1':continue
            
            file_name = f"qa{hop}_{dtype}.jsonl"
            save_dir = f"{save_root}/{file_name}"


            samples = [mkprompt(sub_datas[i], index = i, dtype = dtype) for i in range(400)]


            result_list = vllmpool.submit(
                prompts = samples,
                generate_kwargs  = dict(
                        temperature = 0.7, 
                        max_tokens = 2000, 
                        top_p = 0.9
                )
            )
            result_list.sort(key = lambda x:x['index'])

            for k in result_list:
                k['description'] = k['output']
                k['task'] = f'qa{hop}'
                k.pop('output')
                k.pop('index')

            save_jsonl(result_list, save_dir)
