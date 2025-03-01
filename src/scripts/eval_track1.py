## python scripts/eval_track1.py --folder_name  Meta-Llama-3.1-8B-Instruct
import os, sys, argparse, json
import numpy as np, pandas as pd
from tqdm import tqdm
import nltk.data
from collections import defaultdict
from transformers import pipeline
from loguru import logger
logger.remove()
logger.add(sys.stdout, colorize=True, format="<level>{message}</level>")

sys.path.append(os.path.dirname( os.path.dirname(os.path.abspath(__file__))))
from l_citeeval_metrics import (
    L_cite_eval_cite,
    L_cite_eval_niah_cite,
    L_cite_eval_counting_stars_cite,
    L_cite_eval_counting_stars_cite_zh,
    L_cite_eval_cite_zh,
)

metric1 = {"l_cite_eval_cite":None}
metric2 = {"l_cite_eval_counting_stars_cite":None}
metric3 = {"l_cite_eval_niah_cite":None}
metric4 = {"l_cite_eval_counting_stars_cite_zh":None}
metric5 = {"l_cite_eval_cite_zh":None}
metric_en = {"multihop_qa": metric1,"single_qa":metric1,"counterfact":metric1,'niah': metric3,  'counting_stars':metric2}
metric_zh = {"1_hop": metric5,"2_hop":metric5,"3_hop":metric5,'yes_no': metric5, 'counting_stars':metric4}

METRICS_REGISTRY={
    "l_cite_eval_counting_stars_cite":L_cite_eval_counting_stars_cite,
    "l_cite_eval_cite":L_cite_eval_cite,
    "l_cite_eval_niah_cite":L_cite_eval_niah_cite,
    "l_cite_eval_counting_stars_cite_zh":L_cite_eval_counting_stars_cite_zh,
    "l_cite_eval_cite_zh":L_cite_eval_cite_zh,
}
def get_metric(metric_name):
    return METRICS_REGISTRY[metric_name]
def print_dict_in_table_format(data, excel_file_path):
    benchmark_name_max_len = max(len(name) for name in data.keys())
    task_name_max_len = max(len(task) for tasks in data.values() for task in tasks.keys())
    metric_max_len = max(len(metric) for tasks in data.values() for metrics in tasks.values() for metric in metrics.keys())
    column_widths = [benchmark_name_max_len + 2, task_name_max_len + 10, metric_max_len + 5, 10, 10]
    header = ["LRF(%)", "Tasks", "Metric", "Score", "AVG"]

    logger.info("|{}|{}|{}|{}|{}|".format(
        header[0].center(column_widths[0], ' '),
        header[1].center(column_widths[1], ' '),
        header[2].center(column_widths[2], ' '),
        header[3].center(column_widths[3], ' '),
        header[4].center(column_widths[4], ' ')
    ))
    logger.info("|{}|{}|{}|{}|{}|".format(
        "-" * column_widths[0],
        "-" * column_widths[1],
        "-" * column_widths[2],
        "-" * column_widths[3],
        "-" * column_widths[4]
    ))

    all_scores = []
    rows = []
    task_name_zh = ["1_hop", "2_hop", "3_hop", "yes_no", "counting_stars"]
    task_name_en = ["multihop_qa", "single_qa", "counterfact" ,"counting_stars","niah"]

    for benchmark_name, tasks in data.items():
        benchmark_scores = []
        if "1_hop" in tasks:
            for task in task_name_zh:
                metrics = tasks[task]
                for metric, value in metrics.items():
                    if "f1" not in metric and "acc" not in metric:
                        continue 
                    logger.info("|{}|{}|{}|{}|{}|".format(
                            benchmark_name.center(column_widths[0], ' '),
                            task.center(column_widths[1], ' '),
                            metric.center(column_widths[2], ' '),
                            str(value).center(column_widths[3], ' '),
                            "".center(column_widths[4], ' ')
                        ))
                    rows.append([benchmark_name, task, metric, value, ""])
                    logger.info("|{}|{}|{}|{}|{}|".format(
                        "-" * column_widths[0],
                        "-" * column_widths[1],
                        "-" * column_widths[2],
                        "-" * column_widths[3],
                        "-" * column_widths[4]
                    ))
                    try:
                        score = float(value)
                        benchmark_scores.append(score)
                        all_scores.append(score)
                    except ValueError:
                        logger.warning(f"无法将 {value} 转换为浮点数，跳过该值。")

        else:
            for task in task_name_en:
                metrics = tasks[task]
                for metric, value in metrics.items():
                    if "f1" not in metric and "acc" not in metric:
                        continue 
                    logger.info("|{}|{}|{}|{}|{}|".format(
                            benchmark_name.center(column_widths[0], ' '),
                            task.center(column_widths[1], ' '),
                            metric.center(column_widths[2], ' '),
                            str(value).center(column_widths[3], ' '),
                            "".center(column_widths[4], ' ')
                        ))
                    logger.info("|{}|{}|{}|{}|{}|".format(
                        "-" * column_widths[0],
                        "-" * column_widths[1],
                        "-" * column_widths[2],
                        "-" * column_widths[3],
                        "-" * column_widths[4]
                    ))
                    rows.append([benchmark_name, task, metric, value, ""])
                    try:
                        score = float(value)
                        benchmark_scores.append(score)
                        all_scores.append(score)
                    except ValueError:
                        logger.warning(f"无法将 {value} 转换为浮点数，跳过该值。")


        if benchmark_scores:
            benchmark_avg = round(np.mean(benchmark_scores), 2)
            logger.info("|{}|{}|{}|{}|{}|".format(
                benchmark_name.center(column_widths[0], ' '),
                "Average".center(column_widths[1], ' '),
                "Overall".center(column_widths[2], ' '),
                str(benchmark_avg).center(column_widths[3], ' '),
                "".center(column_widths[4], ' ')
            ))
            rows.append([benchmark_name, "Average", "Overall", benchmark_avg, ""])
        logger.info("|{}|{}|{}|{}|{}|".format(
            "-" * column_widths[0],
            "-" * column_widths[1],
            "-" * column_widths[2],
            "-" * column_widths[3],
            "-" * column_widths[4]
        ))

    if all_scores:
        total_avg = round(np.mean(all_scores), 2)

        logger.info("|{}|{}|{}|{}|{}|".format(
            "Total".center(column_widths[0], ' '),
            "Average".center(column_widths[1], ' '),
            "Overall".center(column_widths[2], ' '),
            "".center(column_widths[3], ' '),
            str(total_avg).center(column_widths[4], ' ')
        ))
        logger.info("|{}|{}|{}|{}|{}|".format(
            "-" * column_widths[0],
            "-" * column_widths[1],
            "-" * column_widths[2],
            "-" * column_widths[3],
            "-" * column_widths[4]
        ))
        rows.append(["Total", "Average", "Overall", "", total_avg])

    # 创建 DataFrame
    df = pd.DataFrame(rows, columns=header)
    

    # 保存到 Excel 文件
    
    df.to_excel(excel_file_path, index=False)
def construct_metrics(metrics_configs,task_name):
    clock = 0
    for metrics_name,metrics_config in metrics_configs.items():
        if not metrics_config:
            metrics_configs[metrics_name] = dict()
            metrics_config = {"test":10}
        if metrics_name in ["l_cite_eval_cite","l_cite_eval_cite_zh"]:
            if clock==0:
                print("If you get stuck here, please check whether the tasksource/deberta-base-long-nli model has been installed for evaluation.")
                clock +=1
            else:
                print()                                                                           
            pipe = pipeline("text-classification",model="tasksource/deberta-base-long-nli", device="cuda:0")
        else:
            pipe = "0";
        metrics_configs[metrics_name]["evaluation"] = get_metric(metrics_name)(pipe=pipe,task_name=task_name,**metrics_config)
    return metrics_configs

def eval():
    benchmark_dict = defaultdict(lambda: defaultdict(dict))
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", type=str, required=True,help="The file at this path should contain data in a specific format,For example")
    args = parser.parse_args()
    args.limit = "auto"
    benchmark_list = []
    folder_name = args.folder_name

    for benchmark_name in os.listdir("generation"):
        benchmark_path = os.path.join("generation", benchmark_name, "prediction",folder_name)
        if os.path.exists(benchmark_path):
            benchmark_list.append(benchmark_name)
    progress_bar = tqdm(benchmark_list)
    logger.info("*"*40+"  evaluating  "+"*"*40)
    for benchmark_name in progress_bar:
        progress_bar.set_description(f"eval benchmark:{benchmark_name}")
        task_list = os.listdir(f"generation/{benchmark_name}/prediction/{folder_name}")
        progress_bar2 = tqdm(task_list)
        for task_name in progress_bar2:

            if task_name.endswith(".json"):
                task_name = task_name[:-5]
            elif task_name.endswith(".jsonl"):
                task_name = task_name[:-6]
            progress_bar2.set_description(f"eval task:{task_name}")
            gathered_metrics = defaultdict(list)

            if benchmark_name == "EN_CiteEval":
                metrics = construct_metrics(metric_en[task_name],task_name)
            else:
                metrics = construct_metrics(metric_zh[task_name],task_name)
            save_task_path = os.path.join("generation",benchmark_name,"track1_results",folder_name,task_name+".jsonl")
            generation_results_path = os.path.join("generation",benchmark_name,"prediction",folder_name,task_name+".jsonl")
            os.makedirs(save_task_path, exist_ok=True)
            if not os.path.exists(generation_results_path):
                continue

            data_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/data/{benchmark_name}/{task_name}.jsonl"
            org_data = {}
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    sample  = json.loads(line.strip())
                    org_data[sample['id']] = sample['passage']
            
            with open(generation_results_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        eval_dict = json.loads(line.strip())
                    except:
                        continue
                    eval_dict["score"] = {}
                    passage, pred, label = org_data[eval_dict['id']], eval_dict["pred"], eval_dict["label"]
                    # org_data[eval_dict["id"]],
                    for metric_name,metric in metrics.items():
                        score = metrics[metric_name]["evaluation"](passage,label, pred)
                        
                        if isinstance(score,dict):
                            for metric_name_sub in score:
                                eval_dict["score"].update(score)
                                gathered_metrics[metric_name_sub].append(score[metric_name_sub])
                        else:
                            eval_dict["score"].update({"metric_name":score})
                            gathered_metrics[metric_name].append(score)
     
            final_metrics = {}
            for metric in gathered_metrics:
                if metric in ["cite_num","niah","cite_num_cite"]:
                    final_metrics[metric] = round(np.array(gathered_metrics[metric]).mean(),2)
                elif metric in ["l_cite_eval_paragraph","l_cite_eval_paragraph_zh"]:
                    final_metrics[metric] = round(np.array(gathered_metrics[metric]).mean(),2)
                else:
                    final_metrics[metric] = round(100*np.array(gathered_metrics[metric]).mean(),2)
            
            benchmark_dict[benchmark_name][task_name] = final_metrics
            logger.info("<<{}>> Final Metric is: {}".format(task_name, final_metrics))

            dump_data = {
                "task_name": task_name,
                "instance_result": gathered_metrics,
                "overall_result": final_metrics,
            }
            with open(
                os.path.join(save_task_path, "final_metrics.jsonl"), "w", encoding="utf-8"
            ) as fout:
                json.dump(dump_data, fout, indent=4, ensure_ascii=False)
                

        output_path = f"./generation/{benchmark_name}/track1_results/{folder_name}"
        os.makedirs(output_path,exist_ok=True)
        print_dict_in_table_format(benchmark_dict,f"./generation/{benchmark_name}/track1_results/{folder_name}/output_table.xlsx")
        logger.info("results_table is saved in {}".format(output_path+"/output_table.xlsx"))
        

if __name__ =="__main__":
    eval()
