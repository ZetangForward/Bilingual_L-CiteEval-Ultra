# Task 6 - Faithful Bilingual Long-context Retrieval and Generation Challenge

[[中文版](README_ZH.md)] [[English](README.md)]

## Task Introduction

In the field of long-context models (LCMs), a key characteristic is the "retrieval-then-generation" paradigm. This involves the model first implicitly identifying key information within the context and then performing generation based on the aggregated context. While significant progress has been made in developing long-context models, i.e., achieving strong performance on open-source long-context benchmarks, there is a need for a more fine-grained and accurate evaluation of their ability to retrieve relevant information and generate faithful outputs. Furthermore, while many open-source LCMs perform well in English-language tasks, their performance in Chinese-language tasks remains unsatisfactory, highlighting a significant gap in the field. To address these challenges, this shared task focuses on assessing two core capabilities of LCMs: retrieval and generation, in bilingual scenarios (Chinese and English). Participants are required to complete the task solely using the LCMs themselves, without relying on external modules like retrieval-augmented generation (RAG). This shared task includes two tracks:

* **Track 1: Long-context Retrieval Faithfulness (LRF)** . Given a query and its corresponding long context, the LCM must explicitly locate and output the necessary key information. This track evaluates the model's ability to accurately identify and extract relevant content from long contexts, assessing its retrieval capability without external assistance. The evaluation will be conducted from two dimensions: fine-grained retrieval (sentence-level) and coarse-grained retrieval (paragraph-level).
* **Track 2: Long-context Generation Faithfulness (LGF)** . This track focuses on the faithfulness of the model's generated outputs. Given a long context that may include information conflicting with real-world knowledge or the model's internal knowledge, such as recent news or updated events, the LCM must strictly adhere to the provided context to generate outputs. This track evaluates the model's ability to generate responses that are both contextually accurate and faithful, without relying on internal knowledge or external retrieval modules. The use of inference-scaling techniques is allowed to enhance the model's performance.

<div align=center>  <img src="Task_Introduction.png" width=80%></div>

## Data Description & Rules

L-CiteEval is a long-context evaluation benchmark, designed to evaluate the information retrieval ability and generation quality of NLP models on long-context tasks, where the model needs to identify critical information while ignoring irrelevent interference.

## Data Format

We build our bilingual evaluation dataset based on [L-CiteEval](https://arxiv.org/abs/2410.02115).
For more construction details, one can directly refer to this paper.

### Chinese

For Chinese dataset, we mainly provide  multi-hop tasks, including **1_hop**, **2_hop** and **3_hop**, each sample of which has interference needles ranging in length from 1 to 16. Based on 1-hop task, we build  **yes_no**  task by adding an answer to the 1_hop question. If the added answer is correct, then the corresponding  answer of the  'yes-no' task is 'yes', otherwise the 'yes-no' task's answer is 'no' . The 'answer-yes' samples and 'answer-no' samples are equally divided in our dataset.

Finally, we add **counting_stars** subset directly from the open source  library [Counting-Stars](https://github.com/nick7nlp/Counting-Stars) , and make sure that its volume is consistent with other subtasks.

<table style="font-size: 16px;" >
  <tr>
    <th>ZH - Task</th><th> Task Name </th><th> Samples</th><th>Length</th> <th> Facts Source </th> <th> Irrlevent Context Source</th></tr>
  <tr><th>qa1</th><th>1_hop</th><th>120</th><th rowspan=5> 8k - 128k </th><th rowspan=4>NLPCC-MH</th><th rowspan=4><a href = https://huggingface.co/datasets/Linly-AI/Chinese-pretraining-dataset>Chinese-Pretraining</a> </th></tr>
<tr><th>qa2</th><th>2_hop</th><th>120</th></tr>
<tr><th>qa3</th><th>3_hop</th><th>120</th></tr>
<tr><th>qa4</th><th>yes_no</th><th>120</th></tr>
<tr><th>qa5</th><th>counting_stars</th><th>120</th><th>-</th> <th> <a href = https://github.com/nick7nlp/Counting-Stars>Counting-Stars<a></th></tr> </table>

### English

For English dataset, we also offer five subtasks. First, we build the **multihop_qa** subtask based on [HotpotQA](https://arxiv.org/pdf/1809.09600)  and [2WikiMultihopQA](https://arxiv.org/pdf/2011.01060) , and build the **single_qa** subtask based on [NarrativeQA](https://arxiv.org/pdf/1712.07040) and [Natural Questions](https://aclanthology.org/Q19-1026.pdf). And for these two tasks, we add additional samples with difficulty levels for users' choices. Then, based on the likeihood that the models may not answer the question according to the provided context, we designed a number of **counterfact** samples, to test the faithfulness of the models to the provided context.

Finally, we add **counting_stars** subset directly from the open source  library [Counting-Stars](https://github.com/nick7nlp/Counting-Stars) , and add **niah** subset from the open source library [NIAH](https://github.com/gkamradt/LLMTest_NeedleInAHaystack/tree/main) .

<table style="font-size: 16px; margin: auto;margin: auto; width: 100%;">
  <tr>
    <th>EN - Task</th><th> Task Name </th><th> Samples</th><th>Length</th> <th> Facts Source </th> <th> Irrlevent Context Source</th></tr>
  <tr><th>qa1</th><th>multihop_qa</th><th>120</th><th rowspan=5> 8k - 64k</th><th><a href = https://huggingface.co/datasets/hotpotqa/hotpot_qa>HotpotQA</a> <br><a href = https://huggingface.co/datasets/voidful/2WikiMultihopQA>2WikiMultihopQA</a></th><th rowspan=3> Remaining Datasets <br> that not used as <br> Facts Source</th></tr>
<tr><th>qa2</th><th>single_qa</th> <th>120</th> <th><a href = https://huggingface.co/datasets/deepmind/narrativeqa> NarrativeQA</a> <br> <a href = https://ai.google.com/research/NaturalQuestions> NaturalQuestions</th></tr>
<tr><th>qa3</th><th>counterfact</th><th>120</th><th>-</th></tr>
<tr><th>qa4</th><th>counting_stars</th><th>120</th><th>-</th><th> <a href = https://github.com/nick7nlp/Counting-Stars>Counting-Stars<a></th></tr>
<tr><th>qa5</th><th>niah</th><th>120</th><th>-</th> <th><a href = https://github.com/gkamradt/LLMTest_NeedleInAHaystack> NIAH</a> </th></tr> </table>

## Data Loading

Basically, you may use the dataset by run :

```python

from datasets import load_dataset

# load Chinese dataset
zh_dataset = load_dataset('ZetangForward/Bilingual_CiteEval', revision="zh")

# load English dataset
en_dataset = load_dataset('ZetangForward/Bilingual_CiteEval', revision="en")


```

## Evaluation

We provide a quick-start evalutaion framework, which mainly evaluate models' ability on metrics:  LRF and LGF.

### Environment Setup

Please download the appropriate verison of flash-attn from  [flash-attn](https://github.com/Dao-AILab/flash-attention/releases) , then run:

```bash
git clone https://github.com/ZetangForward/Bilingual_L-CiteEval-Ultra.git
cd Bilingual_L-CiteEval-Ultra/src
conda create -n citeeval python=3.10 -y
conda activate citeeval
pip install torch==2.5.1
pip install -e .

pip install <path_to_flash_attn_whl_file>
```

### Start Evaluation

Following the environment setup, it's recommended to modify the configuration in **./config/default.yaml** and run the inference script  in the current directory:

```bash
python scripts/run.py  # or export HF_ENDPOINT=https://hf-mirror.com && python scripts/run.py
```

You may also override the default configuration by run:

```bash
python scripts/run.py \
model_path=meta-llama/Llama-3.1-8B-Instruct \
save_tag=Llama-3.1-8B-Instruct \
devices=[0,1] \
tp_size=2
```



After reasoning, you may do evaluating according to your track by run:


```bash
# track 1:
python scripts/eval_track1.py --folder_name <save_tag> # default: python ./scripts/eval_track1.py --folder_name Llama-3.1-8B-Instruct

# track 2:
python scripts/eval_track2.py --folder_name <save_tag> # default: python ./scripts/eval_track2.py --folder_name Llama-3.1-8B-Instruct
```


We present the results of several common models:

<table style="font-size: 16px; margin: auto;margin: auto; width: 85%;">
  <tr>
    <th>Track 1</th>  <th>LRF (%)</th><th>Llama3.1<br>-8B-Instruct</th> <th>Qwen2.5<br>-7B-Instruct</th>
  <th>Mistral-7B<br>-Instruct-v0.3</th><th> glm-4<br>-9b-chat</th></tr>
<tr><th rowspan = 6> ZH </th> <th>1_hop</th> <th>0.19</th><th>2.92</th><th>1.29</th><th>4.33</th></tr>
<tr><th>2_hop</th><th>3.85</th><th>1.64</th><th>0.69</th><th>2.61</th></tr>
<tr><th>3_hop</th><th>2.19</th><th>1.10</th><th>0.78</th><th>3.52</th></tr>
<tr><th>yes_no</th><th>0.00</th><th>3.89</th><th>0.66</th><th>6.52</th></tr>
<tr><th>counting_stars</th><th>5.27</th><th>1.06</th><th>0.28</th><th>4.95</th></tr>
<tr><th>avg.</th><th>2.30</th><th>2.12</th><th>0.74</th><th>4.39</th></tr>
<tr><th rowspan = 6>EN</th><th>multihop_qa</th><th>49.74</th><th>18.14</th><th>18.79</th><th>43.33</th></tr>
<tr><th>single_qa</th><th>28.89</th><th>9.90</th><th>5.20</th><th>30.85</th></tr>
<tr><th>counterfact</th><th>7.22</th><th>12.50</th><th>20.69</th><th>5.20</th></tr>
<tr><th>counting_stars</th><th>22.87</th><th>13.11</th><th>19.51</th><th>23.97</th></tr>
<tr><th>niah</th><th>30.83</th><th>18.06</th><th>12.56</th><th>40.69</th></tr>
<tr><th>avg.</th><th>27.91</th><th>14.34</th><th>15.35</th><th>28.81</th></tr>
<tr><th colspan = 2> AVG.</th><th>15.11</th><th>8.23</th><th>8.05</th><th>16.60</th></tr>


<table style="font-size: 16px; margin: auto;margin: auto; width: 85%;">
  <tr>
    <th>Track 2</th>  <th>LGF (%)</th><th>Llama3.1<br>-8B-Instruct</th> <th>Qwen2.5<br>-7B-Instruct</th>
  <th>Mistral-7B<br>-Instruct-v0.3</th><th> glm-4<br>-9b-chat</th></tr>
<tr><th rowspan = 6> ZH </th> <th>1_hop</th> <th>27.61</th><th>36.46</th><th>16.97</th><th>3.72</th></tr>
<tr><th>2_hop</th><th>10.31</th><th>24.58</th><th>7.66</th><th>2.49</th></tr>
<tr><th>3_hop</th><th>3.87</th><th>10.36</th><th>2.21</th><th>1.05</th></tr>
<tr><th>yes_no</th><th>44.17</th><th>72.50</th><th>32.53</th><th>69.17</th></tr>
<tr><th>counting_stars</th><th>28.18</th><th>44.55</th><th>6.96</th><th>57.59</th></tr>
<tr><th>avg.</th><th>22.83</th><th>37.69</th><th>13.27</th><th>26.80</th></tr>
<tr><th rowspan = 6>EN</th><th>multihop_qa</th><th>14.53</th><th>12.75</th><th>84.59</th><th>4.27</th></tr>
<tr><th>single_qa</th><th>22.21</th><th>18.46</th><th>28.31</th><th>5.49</th></tr>
<tr><th>counterfact</th><th>12.94</th><th>11.49</th><th>14.17</th><th>1.01</th></tr>
<tr><th>counting_stars</th><th>36.25</th><th>57.40</th><th>24.06</th><th>77.92</th></tr>
<tr><th>niah</th><th>93.5</th><th>97.79</th><th>8.76</th><th>96.33</th></tr>
<tr><th>avg.</th><th>35.89</th><th>39.58</th><th>31.98</th><th>37.00</th></tr>
<tr><th colspan = 2> AVG.</th><th>29.36</th><th>38.63</th><th>22.62</th><th>31.90</th></tr></table>

## Training Data Recommendation

Training data is unlimited. The following datasets are recommended:

- [LongAlpaca](https://huggingface.co/datasets/Yukang/LongAlpaca-12k)
- [LongAlign](https://huggingface.co/datasets/THUDM/LongAlign-10k)
- [LongMIT](https://huggingface.co/datasets/donmaclean/LongMIT-128K)
- [Context Synthesis](https://huggingface.co/datasets/Wenhao97/gpt4o-mini-context-synthesis)

## Submission

For submission, the following materials should be packaged as one `zip` file and sent to [zecheng.tang@foxmail.com](zecheng.tang@foxmail.com):

***Submission File** :  After running our evaluation framework, the output will be saved in **./src/generation** , please pack this folder into .zip format and submit this folder. If you use your own evaluation framework, make sure your submission should include the original output of the model and the corresponding evaluation results for all tasks(Track1, Track2 or both).

## Contact & Citation

If your publication employs our dataset, please cite the following article:

```

@article{tang2024citeeval,

  title={L-CiteEval: Do Long-Context Models Truly Leverage Context for Responding?},

  author={Tang, Zecheng and Zhou, Keyan and Li, Juntao and Ji, Baibei and Hou, Jianye and Zhang, Min},

  journal={arXiv preprint arXiv:2410.02115},

  year={2024}

}

```

If you have any questions about this task, please email to **zecheng.tang@foxmail.com**
