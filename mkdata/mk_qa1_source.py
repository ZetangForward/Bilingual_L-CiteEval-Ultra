from datasets import load_dataset

from deploy import *
from mkdata_prompt import *



def mkqa1_prompt(info: dict,task = 'qa', language = 'zh', dtype = 'normal'):
    tuples = info['path']
    for t in tuples:
        for i in range(len(t)):
            t[i] = t[i].split(" ||| ")[0]

    A = tuples[-1][-1]

    return {
        'answer'  : A,
        'fact'   : tuples[-1],
        'input'   : qa1_prompts(tuples[-1],
                            task = task, language = language, 
                            dtype = dtype)
    }




# python mk_qa1_source.py > mk_qa1_source.log 2>&1 &
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
        tp_size = 2,
    )



    datas = load_dataset("/mnt/petrelfs/tangzecheng/NLPCC-MH", split = 'test')    


    qa2 = datas.filter(lambda x:len(x['path'])==2)

    file_name = f"source_qa1.jsonl"
    save_dir = f"{save_root}/{file_name}"


    samples = [mkqa1_prompt(qa2[i]) for i in range(len(qa2))]


    result_list = vllmpool.submit(
        prompts = samples,
        generate_kwargs  = dict(
                temperature = 0.7, 
                max_tokens = 2000, 
                top_p = 0.9
        )
    )

    for k in result_list:
        k['question'] = k['output']
        k.pop('output')

    
    structured_results = []
    for k in result_list:
        path = k['fact']
        path[0] = path[0] + " ||| xxx"
        path[2] = path[2] + " ||| xxx"

        structured_results.append({
            'q':k['question'].split("\n")[1],
            'path': [path]
        })

    save_jsonl(structured_results, save_dir)
