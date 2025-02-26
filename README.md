# Task 6 - Faithful Bilingual Long-context Retrieval and Generation Challenge

## Task Introduction

In the field of long-context models (LCMs), a key characteristic is the "retrieval-then-generation" paradigm. This involves the model first implicitly identifying key information within the context and then performing generation based on the aggregated context. While significant progress has been made in developing long-context models, i.e., achieving strong performance on open-source long-context benchmarks, there is a need for a more fine-grained and accurate evaluation of their ability to retrieve relevant information and generate faithful outputs. Furthermore, while many open-source LCMs perform well in English-language tasks, their performance in Chinese-language tasks remains unsatisfactory, highlighting a significant gap in the field. To address these challenges, this shared task focuses on assessing two core capabilities of LCMs: retrieval and generation, in bilingual scenarios (Chinese and English). Participants are required to complete the task solely using the LCMs themselves, without relying on external modules like retrieval-augmented generation (RAG). This shared task includes two tracks:

* **Track 1: Long-context Retrieval Faithfulness (LRF)** . Given a query and its corresponding long context, the LCM must explicitly locate and output the necessary key information. This track evaluates the model's ability to accurately identify and extract relevant content from long contexts, assessing its retrieval capability without external assistance. The evaluation will be conducted from two dimensions: fine-grained retrieval (sentence-level) and coarse-grained retrieval (paragraph-level).
* **Track 2: Long-context Generation Faithfulness (LGF)** . This track focuses on the faithfulness of the model's generated outputs. Given a long context that may include information conflicting with real-world knowledge or the model's internal knowledge, such as recent news or updated events, the LCM must strictly adhere to the provided context to generate outputs. This track evaluates the model's ability to generate responses that are both contextually accurate and faithful, without relying on internal knowledge or external retrieval modules. The use of inference-scaling techniques is allowed to enhance the model's performance.

<div align=center>  <img src="Task_Introduction.png" width=80%></div>

## Data Description & Rules

L-CiteEval is a novelty benchmark, designed to evaluate the information retrieval ability and generation quality of NLP models on long-context tasks.

L-CiteEval is a novelty benchmark, designed to evaluate the information retrieval ability and generation quality of NLP models on long-context tasks, where the model needs to identify critical information while ignoring irrelevent interference.

To this end, we design a new benchmark construction method in which the dataset for each task undergoes three steps:

(1) **Seed Data & Padding Data Sampling**   (2) **Padding Data Filtering**   (3) **Length Extension**.

We use multiple real and synthetic data sources as a basis, and extend the context length through different strategies to simluate complex retrieval and inference scenarios. The generated test samples' lengths range from 0k to 128k,  aiming to effectively measure the model's long-context comprehension ability.

## Data Format

We provide our bilingual test dataset based on L-CiteEval.

### Chinese

For Chinese dataset, we mainly provide  multi-hop tasks, including **1_hop**, **2_hop** and **3-hop**, each sample of which has interference needles ranging in length from 1 to 16. Based on 1-hop task, we build  **yes_no**  task by adding an answer to the 1_hop question. If the added answer is correct, then the corresponding  answer of the  'yes-no' task is 'yes', otherwise the 'yes-no' task's answer is 'no' . The 'answer-yes' samples and 'answer-no' samples are equally divided in our dataset.

Finally, we add **counting_stars** subset directly from the open source  library [Counting-Stars](https://github.com/nick7nlp/Counting-Stars) , and make sure that its volume is consistent with other subtasks.

<table style="font-size: 14px;">
  <tr>
    <th>ZH - Task</th><th> Task Name </th><th> Samples</th><th>Length</th> <th> Facts Source </th> <th> Irrlevent Context Source</th></tr>
  <tr><th>qa1</th><th>1_hop</th><th rowspan=5>700</th><th rowspan=5>0k - 128k</th><th rowspan=4>NLPCC-MH</th><th rowspan=4><a href = https://huggingface.co/datasets/Linly-AI/Chinese-pretraining-dataset>Chinese-Pretraining</a> </th></tr>
<tr><th>qa2</th><th>2_hop</th></tr>
<tr><th>qa3</th><th>3_hop</th></tr>
<tr><th>qa4</th><th>yes_no</th></tr>
<tr><th>qa5</th><th>counting_stars</th><th>-</th> <th> <a href = https://github.com/nick7nlp/Counting-Stars>Counting-Stars<a></th></tr> </table>

### English

For English dataset, we also offer five subtasks. First, we build the **multihop_qa** subtask based on [HotpotQA](https://arxiv.org/pdf/1809.09600)  and [2WikiMultihopQA](https://arxiv.org/pdf/2011.01060) , and build the **single_qa** subtask based on [NarrativeQA](https://arxiv.org/pdf/1712.07040) and [Natural Questions](https://aclanthology.org/Q19-1026.pdf). And for these two tasks, we add additional samples with difficulty levels for users' choices. Then, based on the likeihood that the models may not answer the question according to the provided context, we designed a small number of **counterfact** samples, to test the faithfulness of the models to the provided context.

Finally, we add **counting_stars** subset directly from the open source  library [Counting-Stars](https://github.com/nick7nlp/Counting-Stars) , and add **NIAH** subtask from the open source [NIAH](https://github.com/gkamradt/LLMTest_NeedleInAHaystack/tree/main) .

<table style="font-size: 14px;">
  <tr>
    <th>EN - Task</th><th> Task Name </th><th> Samples</th><th>Length</th> <th> Facts Source </th> <th> Irrlevent Context Source</th></tr>
  <tr><th>qa1</th><th>multihop_qa</th><th>720</th><th rowspan=5>0k - 128k</th><th><a href = https://huggingface.co/datasets/hotpotqa/hotpot_qa>HotpotQA</a> <br><a href = https://huggingface.co/datasets/voidful/2WikiMultihopQA>2WikiMultihopQA</a></th><th rowspan=3> Remaining Datasets <br> that not used as <br> Facts Source</th></tr>
<tr><th>qa2</th><th>single_qa</th> <th>640</th> <th><a href = https://huggingface.co/datasets/deepmind/narrativeqa> NarrativeQA</a> <br> <a href = https://ai.google.com/research/NaturalQuestions> NaturalQuestions</th></tr>
<tr><th>qa3</th><th>counterfact</th><th>120</th><th>-</th></tr>
<tr><th>qa4</th><th>counting_stars</th><th>360</th><th>-</th><th> <a href = https://github.com/nick7nlp/Counting-Stars>Counting-Stars<a></th></tr>
<tr><th>qa5</th><th>niah</th><th>120</th><th>-</th> <th><a href = https://github.com/gkamradt/LLMTest_NeedleInAHaystack> NIAH</a> </th></tr> </table>

**[NOTE] We pad each en-subset to 720 samples. Please filter out samples with id = -1 when directly using the dataset.**

## Data Loading

Basically, you may use the dataset by run :

```python

from datasets import load_dataset

# load Chinese dataset
zh_dataset = load_dataset('ZetangForward/Bilingual_CiteEval', revision="ZH")

# load English dataset
en_dataset = load_dataset('ZetangForward/Bilingual_CiteEval', revision="EN")


```

## Evaluation

For evaluation, we provide a quick-start evalutaion framework, which evalute models ability on metrics:  precision, recall, f1 and cite numbers.

### Environment Setup

Remeber download the appropriate verison of flash-attn from   [flash-attn](https://github.com/Dao-AILab/flash-attention/releases) , then run:

```bash

git clone https://gitlab.com/iiGray/bilingual_citeeval_benchmark.git
cd Bilingual_L-CiteEval-Ultra
conda create -n citeeval python=3.10 -y
conda activate citeeval
pip install torch==2.5.1
pip install -e .

pip install <path_to_flash_attn_whl_file>
```

### Start Evaluation

It's recommended that modify the configuration in **./config/default.yaml** and run:

```bash
python scripts/run.py  # or export HF_ENDPOINT=https://hf-mirror.com && python scripts/run.py


```

We present the results of several common models:

<table style="font-size: 14px;">
  <tr>
    <th>ZH - Task</th>  <th>Metric</th><th>Llama3<br>-8B-Instruct</th> <th>Llama3.1<br>-8B-Instruct</th>
  <th>Qwen2<br>-7B-Instruct</th><th>Qwen2.5<br>-7B-Instruct</th></tr>
<tr>
    <th rowspan = 3>qa1</th>
<th> f1-answer </th><th><th></th></th><th><th></th></th>
  </tr>
<tr> <th> f1-cite </th> <th></th><th></th></th><th><th></th></tr>
<tr> <th> f1-avg. </th> <th></th><th></th></th><th><th></th></tr>
<tr>
    <th rowspan = 3>qa2</th>
<th> f1-answer </th><th><th></th></th><th><th></th></th>
  </tr>
<tr> <th> f1-cite </th> <th></th><th></th></th><th><th></th></tr>
<tr> <th> f1-avg. </th> <th></th><th></th></th><th><th></th></tr>
<tr>
    <th rowspan = 3>qa3</th>
<th> f1-answer </th><th><th></th></th></th><th><th></th>
  </tr>
<tr> <th> f1-cite </th> <th></th><th></th></th><th><th></th></tr>
<tr> <th> f1-avg. </th> <th></th><th></th></th><th><th></th></tr>
<tr>
    <th rowspan = 3>qa4</th>
<th> f1-answer </th><th><th></th></th></th><th><th></th>
  </tr>
<tr> <th> f1-cite </th> <th></th><th></th></th><th><th></th></tr>
<tr> <th> f1-avg. </th> <th></th><th></th></th><th><th></th></tr>
<tr>
    <th rowspan = 3>qa5</th>
<th> f1-answer </th><th><th></th></th></th><th><th></th>
  </tr>
<tr> <th> f1-cite </th> <th></th><th></th></th><th><th></th></tr>
<tr> <th> f1-avg. </th> <th></th><th></th></th><th><th></th></tr>
<tr> <th colspan = 2>ZH - Avg.</th><th></th><th></th></th><th><th></th></tr>
  <tr>
    <th>EN - Task</th>  <th>Metric</th><th>Llama3<br>-8B-Instruct</th> <th>Llama3.1<br>-8B-Instruct</th>
  <th>Qwen2<br>-7B-Instruct</th><th>Qwen2.5<br>-7B-Instruct</th></tr>
<tr>
    <th rowspan = 3>qa1</th>
<th> f1-answer </th><th><th></th></th></th><th><th></th>
  </tr>
<tr> <th> f1-cite </th> <th></th><th></th></th><th><th></th></tr>
<tr> <th> f1-avg. </th> <th></th><th></th></th><th><th></th></tr>
<tr>
    <th rowspan = 3>qa2</th>
<th> f1-answer </th><th><th></th></th></th><th><th></th>
  </tr>
<tr> <th> f1-cite </th> <th></th><th></th></th><th><th></th></tr>
<tr> <th> f1-avg. </th> <th></th><th></th></th><th><th></th></tr>
<tr>
    <th rowspan = 3>qa3</th>
<th> f1-answer </th><th><th></th></th></th><th><th></th>
  </tr>
<tr> <th> f1-cite </th> <th></th><th></th></th><th><th></th></tr>
<tr> <th> f1-avg. </th> <th></th><th></th></th><th><th></th></tr>
<tr>
    <th rowspan = 3>qa4</th>
<th> f1-answer </th><th><th></th></th></th><th><th></th>
  </tr>
<tr> <th> f1-cite </th> <th></th><th></th></th><th><th></th></tr>
<tr> <th> f1-avg. </th> <th></th><th></th></th><th><th></th></tr><tr>
    <th rowspan = 3>qa5</th>
<th> f1-answer </th><th><th></th></th></th><th><th></th>
  </tr>
<tr> <th> f1-cite </th> <th></th><th></th></th><th><th></th></tr>
<tr> <th> f1-avg. </th> <th></th><th></th></th><th><th></th></tr>
<tr> <th colspan = 2>EN - Avg.</th><th></th><th></th></th><th><th></th></tr>
<tr> <th colspan = 2> <b> AVG. <b></th><th></th><th></th></th><th><th></th></tr>
</table>

## Submission

For submission, the following materials should be packaged as one `zip` file and sent to [xzs23@mails.tsinghua.edu.cn](mailto:xzs23@mails.tsinghua.edu.cn):

***Submission File** : The output sentences should be written into one text file. **The format of the submission file must be the same as the input file. Specifically, the submission file must contain the same number of lines as the input file, and each line is a correct sentence corresponding to the sentence in the input file.**

***Code** : The code folder should contain all the codes of data augmentation, data processing, model training, and model inference.

***Document** :

***Data Description** : The document needs to contain a brief description of supervised and unsupervised data used in the experiment, as well as the data augmentation methods for unsupervised data.

***Sharing Link of Unsupervised Data** : Unsupervised data used in the experiment should be uploaded to a cloud storage, i.e., net disk, and the sharing link should be included in the document. It is not allowed to use data that violates the rules during model training.

## Contact & Citation

If your publication employs our dataset, please cite the following article:

**复制**

```

@article{tang2024citeeval,

  title={L-CiteEval: Do Long-Context Models Truly Leverage Context for Responding?},

  author={Tang, Zecheng and Zhou, Keyan and Li, Juntao and Ji, Baibei and Hou, Jianye and Zhang, Min},

  journal={arXiv preprint arXiv:2410.02115},

  year={2024}

}

```

If you have any questions about this task, please email to xxx
