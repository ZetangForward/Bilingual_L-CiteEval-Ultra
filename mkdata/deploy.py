import os
from typing import List
import vllm, json
from tqdm import tqdm
import multiprocessing as mp


def save_jsonl(list_data: list, path: str):
    with open(path, "w") as f:
        for data in list_data:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")


class QuickVllm:
    def __init__(self,
                 model_name = "meta-llama/Meta-Llama-3-8B-Instruct",
                 generate_kwargs: dict = {}):
        self.model_name = model_name
        self.generate_params = vllm.SamplingParams(**generate_kwargs)
        self.model = vllm.LLM(model = model_name,
                              trust_remote_code = True,
                              gpu_memory_utilization = 0.97)

    def chat(self, conversation):
        outputs = self.model.chat(conversation)

        return outputs[0].outputs[0].text


class VllmPoolExecutor:
    def __init__(self,
                 model_name = "meta-llama/Meta-Llama-3-8B-Instruct",
                 gpu_list = [0, 1, 2, 3, 4, 5, 6, 7],
                 tp_size = 2):

        self.model_name = model_name
        self.gpu_list = list(map(str, gpu_list))
        self.tp_size = tp_size
        self.gpus = [",".join(self.gpu_list[i:i+tp_size]) \
                     for i in range(0, len(gpu_list), tp_size)]
        self.manager = mp.Manager()

    def chunks(self, lst, chunk_num):
        chunk_width = len(lst) // chunk_num
        ones = chunk_num - len(lst) % chunk_num 
        p = 0
        for i in range(chunk_num):
            if i == ones: chunk_width += 1
            yield lst[p: p + chunk_width]
            p += chunk_width

    @classmethod
    def worker(cls, visible_gpus, model_name, generate_kwargs, prompts, result_list):
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus
        qvllm = QuickVllm(model_name, generate_kwargs)

        for prompt in tqdm(prompts):
            output = qvllm.chat(prompt['input'])
            prompt.pop('input')
            prompt['output'] = output
            result_list.append(prompt)



    def submit(self,prompts:List[dict], 
               generate_kwargs :dict = dict(
                   temperature = 0.7, 
                   max_tokens = 100, 
                   top_p = 0.95)
                   ):

        result_list = self.manager.list()
        
        chunked_prompts = self.chunks(prompts, len(self.gpus))
        processes = []
        for visible_gpus, chunked_prompt in zip(self.gpus, chunked_prompts):
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus
            process = mp.Process(
                target = self.worker,
                args = (
                    visible_gpus,
                    self.model_name,
                    generate_kwargs,
                    chunked_prompt,
                    result_list
                )
            )
            process.start()
            processes.append(process)
        
        for p in processes:
            p.join()


        return list(result_list)




# nohup python deploy.py > log 2>&1 &
if __name__ == "__main__":
    

    vllmpool = VllmPoolExecutor(model_name = "meta-llama/Meta-Llama-3-8B-Instruct",
                                gpu_list = [0, 1], 
                                tp_size = 1)
    
    print(vllmpool.submit(
        prompts = [
            {   
                'question': 'Write an essay about the importance of higher education.',
                'input':[{"role": "system","content": "You are a helpful assistant"},
                         {"role": "user","content": "Hello"},
                         {"role": "assistant", "content": "Hello! How can I assist you today?"},
                         {"role": "user", "content": "Write an essay about the importance of higher education."}],
            },
            {   'question': 'What is the apple?',
                'input':[{"role": "system","content": "You are a helpful assistant"},
                         {"role": "user","content": "Hello"},
                         {"role": "assistant", "content": "Hello! How can I assist you today?"},
                         {"role": "user", "content": "What is the apple?"}],
                         }
        ],
        generate_kwargs = dict(
            temperature = 0.7, 
            max_tokens = 100, 
            top_p = 0.95,
    ),    
    ))