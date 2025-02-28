import os, sys, time, random, hydra, subprocess, torch
import numpy as np, torch.multiprocessing as mp
from loguru import logger
logger.remove()
logger.add(sys.stdout, colorize=True, format="<level>{message}</level>")


from prepare import DataLoader
from pred import Evaluator

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

def format_localtime():
    return time.strftime("%mM_%dD_%HH_%Mm", time.localtime())


def seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg):
    mp.set_start_method('spawn')
    model_name = cfg.model_path.split("/")[-1]

    save_tag = f"{model_name}_{format_localtime()}" \
        if not cfg.save_tag else cfg.save_tag

    if len(cfg.devices) == 0:
        gpu_count = torch.cuda.device_count()
        cfg.devices = ','.join(map(str, range(gpu_count)))

    start_time = time.time()
    assert len(cfg.devices) % cfg.tp_size==0, \
        "The number of GPUs must be divided evenly with a appropriate tp_size."

    seed(cfg.random_seed)


    DataLoader(cfg.benchmarks, getattr(cfg,"limit",float("inf"))).load()   
 
    #start to generate
    evaluator = Evaluator(cfg, save_tag)
    evaluator.run(cfg.tp_size)
    
    logger.info(f"All generated data has been successfully stored in generation/")
    #eval
    if cfg.do_eval:
        command = ["python","./scripts/eval.py",
                    "--folder_name",f"{save_tag}"]
        subprocess.run(command)

    #execution_time
    execution_time = time.time()-start_time
    logger.info("The total running time was : {:02d}:{:02d}:{:02d}".format(int(execution_time // 3600), int((execution_time % 3600) // 60), int(execution_time % 60)))

if __name__ == "__main__":
    main()