from modelzipper.tutils import *
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.qa_dataset import TaskDataset, BackgroundSampler
from data.counting_star_utils import CountingStarDataset

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")

data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data")
print(data_path)
background_sampler = BackgroundSampler(tokenizer, language = 'zh', root_dir=data_path)

datasets = ["qa1", "qa2", "qa3", "counting_stars", "yes/no"]


qa_dicts = {}

for task_type in tqdm(['qa1','qa2','qa3','yes/no']):
    ret = []
    for context_length in tqdm([0, 4000, 8000, 16000, 32000, 64000, 128000]):

        taskset = TaskDataset(tokenizer = tokenizer,
                              background_sampler = background_sampler,
                            tokenize=False,
                            task = task_type,
                            context_length = context_length,
                            dataset_length = 100)
        if task_type =='yes/no':ll = 100
        else: ll =100
        for i in tqdm(range(ll)):
            ret.append(taskset[i])
            ret[-1].pop('evidence_source')
            ret[-1]['context_length'] = f"{context_length//1000}k"
            ret[-1]['task'] = task_type

    qa_dicts[task_type] = Dataset.from_list(ret)
    



counting_star_result = []
for context_length in tqdm([0, 4000, 8000, 16000, 32000, 64000, 128000]):
    testset = CountingStarDataset(
        context_length = context_length,
        dataset_length = 100,
        language = 'zh',
        root_dir = data_path,
    )
    
    for i in tqdm(range(len(testset))):
        counting_star_result.append(testset[i])
        counting_star_result[-1].pop("index")
        counting_star_result[-1].pop("language")
        counting_star_result[-1]['context_length'] = f"{context_length//100}k"
        counting_star_result[-1]['task'] = f"counting_star-{counting_star_result[-1]['task_type'].split('-')[-1]}"
        counting_star_result[-1].pop("task_type")

counting_stars_dataset = Dataset.from_list(counting_star_result)


new_dict = {
    **qa_dicts,
    "counting_stars": counting_stars_dataset
}

new_dataset = Dataset.from_dict(new_dict)

new_dataset = DatasetDict({"zh_citeeval": new_dataset})
print(new_dataset)
# exit(0)

new_dataset.push_to_hub("ZetangForward/ZH_CiteEval", private=True, token="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")