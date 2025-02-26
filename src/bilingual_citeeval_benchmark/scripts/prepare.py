import os, shutil
import json
from typing import Literal, List, Dict
from tqdm import tqdm
from loguru import logger
from datasets import load_dataset

def format_tasks(all_tasks):
    formatted_tasks = ""
    l = 0
    for key, value in all_tasks.items():
        formatted_tasks+=f"{len(value)} tasks: from "
        formatted_tasks += f"{key}:{value} "
        l += len(value)
        formatted_tasks += f"\n"
    formatted_tasks += f"Totally {l} tasks"
    return formatted_tasks
def transform_zh(raw_data,task_name):
    if task_name=="counting_stars":
        if raw_data["context_length"]=="0k":
            raw_data["passage"] = raw_data["question"].split("\n")
        else:
            raw_data["passage"] = raw_data["question"].split("。")
    else:
        raw_data["passage"] = (raw_data["context"]+raw_data["question"]).split("。")
    raw_data["choices"] = ""
    raw_data["label"] = raw_data["answer"]

    
    return raw_data
def transform_en(raw_data):
    raw_data["passage"] = raw_data["docs"]
    raw_data["choices"] = ""
    raw_data["label"] = raw_data["answer"]
    raw_data.pop("answer")
    raw_data.pop("docs")
    return raw_data

def make_data(dataset, benchmark_name, task_name, limit):
    output_path = f"./data/{benchmark_name}/{task_name}.jsonl"
    os.makedirs(f"./data/{benchmark_name}", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for index, raw_data in enumerate(dataset):
            if benchmark_name=="EN_CiteEval":
                if raw_data["id"]==-1 and raw_data["length"]==-1:
                    continue
            if index>limit:
                break
            if benchmark_name=="EN_CiteEval":
                new_data = transform_en(raw_data)
            else:
                new_data = transform_zh(raw_data,task_name)
            fout.write(json.dumps(new_data, ensure_ascii=False) + "\n")

class DataLoader:
    def __init__(self, benchmarks: Dict[str, List[str]], samples_per_task: int):
        
        self.benchmarks = benchmarks
        self.samples_per_task =samples_per_task
        print(benchmarks)
        formatted_output = format_tasks(benchmarks)
        logger.info(f"The tasks you've selected are as follows:\n{formatted_output}")
        logger.info("Benchmark data is currently being downloaded and transformed...")
    

    def load(self):
        tasks_path_list = [] 
        progress_bar = tqdm(self.benchmarks.items())
        for benchmark_name, task_names in progress_bar:   
            progress_bar.set_description(f"Downloading and transforming {benchmark_name} data")
            data = load_dataset('ZetangForward/Bilingual_CiteEval',revision=benchmark_name[:2],cache_dir="./data/tmp_Rawdata",trust_remote_code=True,download_mode="reuse_cache_if_exists")

            for task_name in task_names:
                progress_bar.set_description(f"Downloading and transforming task {task_name}")
                dataset = data[benchmark_name.lower()][task_name]
                make_data(dataset, benchmark_name, task_name, limit = self.samples_per_task)

                task_path = f"data/{benchmark_name}/{task_name}.jsonl"
                tasks_path_list.append(task_path)
        
        logger.info(f"All generated data has been successfully stored in: {os.path.abspath('./data/')}")
        return tasks_path_list