from datasets import load_dataset

from deploy import *
from mkdata_prompt import *



def mkyesno_prompt(info: dict,index, yesno ,task = 'qa', language = 'zh',dtype = 'normal'):

    return {
        'origin_answer'  : info['answer'],
        'answer'   : yesno,
        'facts'   :  info['facts'],
        'description': info['description'],
        'input'   : yesno_prompts(
                            question(info['question']) + answer(info['answer']) + prompt(yesno),
                            task = task, language = language, 
                            dtype = dtype),
        'index' :index, 
    }




# python mk_yesno_source.py > mk_yesno_source.log 2>&1 &
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




    qa1 = read_jsonl("/mnt/petrelfs/tangzecheng/Bilingual_L-CiteEval-Ultra/data/zh/qa1_short.jsonl")

    file_name = f"yesno_source.jsonl"
    save_dir = f"{save_root}/{file_name}"


    samples = [mkyesno_prompt(qa1[i],index = i, yesno = '是' if i&1 else '否') for i in range(len(qa1))]


    result_list = vllmpool.submit(
        prompts = samples,
        generate_kwargs  = dict(
                temperature = 0.7, 
                max_tokens = 2000, 
                top_p = 0.9
        )
    )

    result_list.sort(key= lambda x:x['index'])
    for k in result_list:
        k['question'] = k['output']
        k.pop('output')
        k.pop('index')



    save_jsonl(result_list, save_dir)
