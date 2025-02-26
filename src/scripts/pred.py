
import os, sys, json, random, time, torch
import torch.multiprocessing as mp
from tqdm import tqdm
from loguru import logger
logger.remove()
logger.add(sys.stdout, colorize=True, format="<level>{message}</level>")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models_deploy import get_model
from utils.instance import Instance
from utils.request import Request
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
prompt_zh={  
        "1_hop":"{D}\n",
        "2_hop":"{D}\n",
        "3_hop":"{D}\n",
        "yes_no":"{D}\n",
        "counting_stars":"{D}\n"}
prompt_en = {
        "multihop_qa":"{D}\n\nWrite an accurate, engaging, and concise answer to the given question using only the provided passages (some of which might be irrelevant). Use an unbiased and journalistic tone. Every sentence must include a citation at the end, referencing at least one passage and at most three. When citing several passages, use separate brackets for each index number, like [a][b][c], instead of combining them in one set of brackets, like [a, b, c]. Here, a, b and c represent different index numbers. If multiple passages support the sentence, only cite a minimum sufficient subset of the passages.\n\nQuestion: {Q}\nAnswer: ",
        "counting_stars":"{D}\n\nOn this moonlit and misty night, the little penguin is looking up at the sky and concentrating on counting \u2605. Please help the little penguin collect the correct number of \u2605 and cite the corresponding passage ID where the counting is mentioned, for example: {{'little_penguin': [x, x, x,...], 'passage_id': [y, y, y,...]}}. The summation is not required. The numbers in [x, x, x,...] represent the correctly counted number of \u2605 by the little penguin and the number in [y, y, y,...] represent the passage IDs where these counts are recorded. Only output the results in JSON format without any explanation.\nAnswer:",
        "single_qa":"{D}\n\nWrite an accurate, engaging, and concise answer to the given question using only the provided passages (some of which might be irrelevant). Use an unbiased and journalistic tone. Every sentence must include a citation at the end, referencing at least one passage and at most three. When citing several passages, use separate brackets for each index number, like [a][b][c], instead of combining them in one set of brackets, like [a, b, c]. Here, a, b and c represent different index numbers. If multiple passages support the sentence, only cite a minimum sufficient subset of the passages.\n\nQuestion: {Q}\nAnswer:",
        "counterfact":"{D}\n\nWrite an accurate, engaging, and concise answer to the given question using only the provided passages (some of which might be irrelevant). Use an unbiased and journalistic tone. Every sentence must include a citation at the end, referencing at least one passage and at most three. When citing several passages, use separate brackets for each index number, like [a][b][c], instead of combining them in one set of brackets, like [a, b, c]. Here, a, b and c represent different index numbers. If multiple passages support the sentence, only cite a minimum sufficient subset of the passages.\n\nQuestion: {Q}\nAnswer:",
        "niah":"{D}\n\nWrite an accurate, engaging, and concise answer to the given question using only the provided passages (some of which might be irrelevant). Use an unbiased and journalistic tone. Every sentence must include a citation at the end, referencing at least one passage and at most three. When citing several passages, use separate brackets for each index number, like [a][b][c], instead of combining them in one set of brackets, like [a, b, c]. Here, a, b and c represent different index numbers. If multiple passages support the sentence, only cite a minimum sufficient subset of the passages.\n\nQuestion: {Q}\nAnswer:",
    }
llm_param1 = {"max_tokens": 128,"temperature": 0,"top_p": 1,"stop":"\n","do_sample":False}
llm_param2 = {"max_tokens": 200,"temperature": 0,"top_p": 1,"stop":"\n","do_sample":False}
llm_param3 = {"max_tokens": 800,"temperature": 0,"top_p": 1,"stop":"\n","do_sample":False}
llm_params = {"multihop_qa":llm_param2,"single_qa": llm_param2,"counterfact": llm_param2,'niah':llm_param1,  'counting_stars':  llm_param1,"1_hop": llm_param2,"2_hop": llm_param2,"3_hop": llm_param2,"yes_no": llm_param1}


def transform(data,task_name,benchmark_name):
    if benchmark_name=="EN_CiteEval":
        for prompt in prompt_en:
            if prompt in task_name:
                with open("./demo_prompt/EN_CiteEval/{}".format(prompt+"_default.jsonl"), 'r') as f:
                    demo_prompt = json.load(f)
                model_input= get_instruction_template(prompt, demo_prompt, data)
                return model_input
    else:
        for prompt in prompt_zh:
            if prompt in task_name:

                with open("./demo_prompt/ZH_CiteEval/{}".format(prompt+"_default.jsonl"), 'r') as f:
                    demo_prompt = json.load(f)
                model_input= get_instruction_template(prompt, demo_prompt, data)
                return model_input
    
def make_doc_prompt(doc, doc_id, doc_prompt):
    if type(doc) == str:
        text = doc
    elif type(doc) == dict:
        if 'title' in doc:
            title = doc['title']
            text = doc['text'].strip('\n')
            if text[:len(title)+1] == title + '\n':
                text = text[len(title)+1:]
        else:
            text = doc['text'].strip('\n')

    return doc_prompt.replace("{P}", text).replace("{ID}", str(doc_id+1))


def make_demo(item, prompt, ndoc=None, doc_prompt=None, instruction=None, test=False):

    if "{Q}" in prompt:
        prompt = prompt.replace("{INST}", instruction).replace("{Q}", item['question'])
    else:
        prompt = prompt.replace("{INST}", instruction)
    if "{D}" in prompt:
        doc_list = item["docs"]
        text = "".join([make_doc_prompt(doc, doc_id, doc_prompt) for doc_id, doc in enumerate(doc_list)])
        prompt = prompt.replace("{D}", text)
        
    answer = "\n" + "\n".join(item["answer"]) if isinstance(item["answer"], list) else item["answer"]
    prompt = prompt.replace("{A}", "").rstrip() + answer
    return prompt

def make_demo2(data, prompt,ndoc=None, doc_prompt=None, instruction=None, test=False):

    if "{Q}" in prompt:
        prompt = prompt.replace("{INST}", instruction).replace("{Q}", data["question"])
    else:
        prompt = prompt.replace("{INST}", instruction)
    if "{D}" in prompt:
        doc_list = data["passage"]

        text = "".join([make_doc_prompt(doc, doc_id, doc_prompt) for doc_id, doc in enumerate(doc_list)])

        prompt = prompt.replace("{D}", text)
    prompt = prompt.replace("{A}", "").rstrip() 
    return prompt

def get_instruction_template(task, prompt, sample):

    head_prompt = ""
    if task in ["dialsim"]:      
        head_prompt += make_demo(
            prompt['demos'][0], prompt=prompt["demo_prompt"], doc_prompt=prompt["doc_prompt"], instruction=prompt["instruction"].replace("<<<chatbox>>>", prompt['demo_role'])
        )
    else:
        head_prompt += make_demo(
            prompt['demos'][0], prompt=prompt["demo_prompt"], doc_prompt=prompt["doc_prompt"], instruction=prompt["instruction"]
        )
    head_prompt += prompt["demo_sep"]

    if task in ["dialsim"]:  
        head_prompt += make_demo2(
            sample, prompt["demo_prompt"] ,doc_prompt=prompt["doc_prompt"],
            instruction=prompt["instruction"].replace("<<<chatbox>>>", "Sheldon"), test=True
        )
    else:
        head_prompt += make_demo2(
            sample, prompt["demo_prompt"],doc_prompt=prompt["doc_prompt"],
            instruction=prompt["instruction"], test=True
        )
    return head_prompt

class Evaluator:
    def __init__(self, args, file_name):
        #Set parameters
        self.tasks_list = []
        self.args = args
        self.file_name = file_name
        self.limit = int(args.limit) if args.limit!="auto" else 10000
        self.build_tasks(args.benchmarks) #Create a task based on the configuration file.

    def build_tasks(self, benchmarks):
        for benchmark_name, task_names in benchmarks.items():
            for task_name in task_names:
                path = f"./data/{benchmark_name}/{task_name}.jsonl"
                with open(path, "r", encoding="utf-8") as file:
                    for index, line in enumerate(file):
                        if index>=self.limit:
                            break
                        raw_input = Instance(json.loads(line.strip()))
                        prompt_input = transform(raw_input.data, task_name,benchmark_name)
                        self.tasks_list.append([task_name,benchmark_name, Request(
                            prompt_input=prompt_input,
                            params=llm_params[task_name],
                            raw_example=raw_input,
                        )])

    def set_cuda_visible_devices(self, devices: str):
        os.environ["CUDA_VISIBLE_DEVICES"] = devices

    def chunks(self, lst: list, chunk_num: int):
        chunk_width = len(lst) // chunk_num
        ones = chunk_num - len(lst) % chunk_num 
        p = 0
        for i in range(chunk_num):
            if i == ones: chunk_width += 1
            yield lst[p: p + chunk_width]
            p += chunk_width

    def run(self, tp_size):
        random.shuffle(self.tasks_list)
        gpus=list(map(str,self.args.devices))
        devices_list = [",".join(gpus[i: i + tp_size]) \
                        for i in range(0, len(gpus), tp_size)]
        processes = []
        for i , (raw_data, devices) in enumerate(zip(
            self.chunks(self.tasks_list, chunk_num = len(gpus)//tp_size), devices_list 
        )):
            self.set_cuda_visible_devices(devices)
            p = mp.Process(
                target=self.get_pred, 
                args=(i, raw_data, devices)
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


    def get_pred(self, i, raw_data, devices):
        #model depoly     
        model = get_model(self.args.server)(self.args,devices)
        model.deploy()
        failed = 0
        print("LEN:",len(raw_data))
        for task_name, benchmark_name, request in tqdm(raw_data, desc=f"Rank: {i}"):

            prompt = request.prompt_input
            if self.args.template:
                prompt = self.args.template.format(user_input=prompt, assistant_response='')
            elif hasattr(model.tokenizer, 'apply_chat_template') and hasattr(model.tokenizer, 'chat_template') and model.tokenizer.chat_template:
                prompt = model.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True
            )
            request.prompt_input = prompt
            try:
                with torch.no_grad(): 
                    result = model.generate(request.params, request.prompt_input)
            except Exception as general_err:
                failed += 1
                print(f"An unexpected error occurred: {general_err}. Total failed runs: {failed} **********")
            if "答案:" in result:
                result = result.replace("答案:","")
            if "answer:" in result:
                result = result.replace("answer:","")
            if "答案：" in result:
                result = result.replace("答案：","")
            path = os.path.join("generation",benchmark_name,"prediction",f"{self.file_name}",task_name+".jsonl")
            os.makedirs(os.path.join("generation",benchmark_name,"prediction",f"{self.file_name}"), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                data = request.raw_example.data
                data["choices"] = data["choices"]
                data["pred"] = result
                data["model_input"] = request.prompt_input
                json.dump(data, f, ensure_ascii=False)
                f.write('\n')
