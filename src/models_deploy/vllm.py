import time
import os
import torch
from copy import deepcopy
import sys
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
from loguru import logger
logger.remove()
logger.add(sys.stdout,
        colorize=True, 
        format="<level>{message}</level>")
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams,TokensPrompt
import time

class Vllm():
    def __init__(self,
                 args, 
                 devices,
                 params_dict = {
                     "max_tokens": 200,
                }):
        self.args = args
        self.devices = devices
        self.params_dict = params_dict
    def deploy(self):
        time_start = time.time()
        self.tokenizer =  AutoTokenizer.from_pretrained(self.args.model_path,
                                                        trust_remote_code = True,
                                                        mean_resizing=False)
        # Load the model and tokenizer
        if hasattr(self.args, "max_model_len") and self.args.max_model_len:
            self.model = LLM(
                model=self.args.model_path,
                trust_remote_code = True,
                tensor_parallel_size=len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")),
                gpu_memory_utilization=0.97,
                enforce_eager=True,
                max_model_len = self.args.max_model_len
                )
        else:
            self.model = LLM(
                model=self.args.model_path,
                trust_remote_code = True,
                tensor_parallel_size=len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")),
                gpu_memory_utilization=0.97,
                enforce_eager=True)
        # Check and add pad token if necessary
        logger.info("Model and tokenizer initialized.",flush=True )
        time_cost = time.time()-time_start
        logger.info("Model_deploy time :{}".format(time_cost),flush=True )

    def generate(self,params_dict,prompt):
        params_ = deepcopy(self.params_dict)
        params_.update(params_dict)
        del_list = ["do_sample","num_beams"]
        for i in del_list:
            if i in params_dict:
                params_dict.pop(i)

        outputs = self.model.generate(prompt,SamplingParams(**params_dict))


        generated_text = outputs[0].outputs[0].text
        res = generated_text

        torch.cuda.empty_cache()
        return res











  