# Task 6 - 双语长上下文检索与生成的忠实度挑战

[[中文版](README_ZH.md)] [[English](README.md)]

## 任务介绍

在长上下文模型 (LCM) 领域，一个关键特征是“先检索后生成”范式。模型首先隐式识别上下文中的关键信息，然后基于聚合的上下文进行生成。虽然开发长文本模型已经取得了重大进展，即模型在开源长文本基准上取得了出色的表现，但仍需要更细粒度和准确地评估它们检索相关信息和生成忠实输出的能力。此外，虽然许多开源 LCM 在英语任务中表现良好，但它们在中文任务中的表现仍然不令人满意，凸显了该领域的重大差距。为了应对这些挑战，此共享任务侧重于评估 LCM 在双语场景（中文和英文）中的两个核心功能：检索和生成。参与者需要仅使用 LCM 本身完成任务，而不依赖于检索增强生成 (RAG) 等外部模块。此共享任务包括两个Track：

* **Track 1：长上下文检索忠实度（LRF）** 。给定一个问题及其对应的长上下文，LCM 必须明确地定位并输出必要的关键信息。此 Track 评估模型从长上下文中准确识别和提取相关内容的能力，及评估其在没有外部帮助的情况下的检索能力。评估将从两个维度进行：细粒度检索（句子级别）和粗粒度检索（段落级别）。
* **Track 2：长上下文生成忠实度 (LGF)** 。此Track侧重于模型生成输出的忠实度。给定一个可能包含与现实世界知识或模型内部知识相冲突的信息的长上下文，例如最近的新闻或更新的事件，LCM 必须严格遵守提供的上下文来生成输出。该Track评估模型在不依赖内部知识或外部检索模块的情况下生成上下文准确且忠实的响应的能力。允许使用推理缩放技术来增强模型的性能。

<div align=center>  <img src="Task_Introduction.png" width=80%></div>

## 数据描述与规则

L-CiteEval 是一个长文本评测基准，旨在评估NLP模型在长上下文任务中的信息检索能力和生成质量，其中模型需要在忽略无关干扰的同时识别关键信息。

## 数据格式

我们基于 [L-CiteEval](https://arxiv.org/abs/2410.02115) 提供了双语测试数据集。
关于更多的数据构建细节，可直接参考论文（对于本次的share task，测试数据的构建细节并不重要，下面将提供评测数据的数据格式）。

### 中文

对于中文数据集，我们主要提供多跳任务，包括 **1_hop**，**2_hop** 和 **3_hop**，每个样本的干扰项长度范围从 1 到 16。基于 1-hop 任务，我们通过为 1_hop 问题添加答案构建 **yes_no** 任务。如果添加的答案正确，那么对应的 'yes-no' 任务答案为 'yes'，否则 'yes-no' 任务答案为 'no'。我们的数据集中，'answer-yes' 样本和 'answer-no' 样本的数量是平衡的。

最后，我们直接从开源库 [Counting-Stars](https://github.com/nick7nlp/Counting-Stars) 中添加 **counting_stars** 子集，并确保其数据量与其他子任务一致。

<table style="font-size: 16px;" >
  <tr>
    <th>ZH - Task</th><th> Task Name </th><th> Samples</th><th>Length</th> <th> Facts Source </th> <th> Irrlevent Context Source</th></tr>
  <tr><th>qa1</th><th>1_hop</th><th>120</th><th rowspan=5> 8k - 128k </th><th rowspan=4><a href = https://github.com/wavewangyue/NLPCC-MH>NLPCC-MH<a></th><th rowspan=4><a href = https://huggingface.co/datasets/Linly-AI/Chinese-pretraining-dataset>Chinese-Pretraining</a> </th></tr>
<tr><th>qa2</th><th>2_hop</th><th>120</th></tr>
<tr><th>qa3</th><th>3_hop</th><th>120</th></tr>
<tr><th>qa4</th><th>yes_no</th><th>120</th></tr>
<tr><th>qa5</th><th>counting_stars</th><th>120</th><th>-</th> <th> <a href = https://github.com/nick7nlp/Counting-Stars>Counting-Stars<a></th></tr> </table>

### 英文

对于英文数据集，我们也提供了五个子任务。首先，我们基于 [HotpotQA](https://arxiv.org/pdf/1809.09600) 和 [2WikiMultihopQA](https://arxiv.org/pdf/2011.01060) 构建了 **multihop_qa** 子任务，并基于 [NarrativeQA](https://arxiv.org/pdf/1712.07040) 和 [Natural Questions](https://aclanthology.org/Q19-1026.pdf) 构建了 **single_qa** 子任务。然后，对于模型可能不会根据提供的上下文回答问题的情况，我们设计了一些 **counterfact** 样本，以测试模型对提供上下文的忠实度。

最后，我们直接从开源库 [Counting-Stars](https://github.com/nick7nlp/Counting-Stars) 中添加了 **counting_stars** 子集，并从开源项目 [NIAH](https://github.com/gkamradt/LLMTest_NeedleInAHaystack/tree/main) 中添加了 **niah** 子任务。

<table style="font-size: 16px; margin: auto;margin: auto; width: 100%;">
  <tr>
    <th>EN - Task</th><th> Task Name </th><th> Samples</th><th>Length</th> <th> Facts Source </th> <th> Irrlevent Context Source</th></tr>
  <tr><th>qa1</th><th>multihop_qa</th><th>120</th><th rowspan=5> 8k - 64k</th><th><a href = https://huggingface.co/datasets/hotpotqa/hotpot_qa>HotpotQA</a> <br><a href = https://huggingface.co/datasets/voidful/2WikiMultihopQA>2WikiMultihopQA</a></th><th rowspan=3> Remaining Datasets <br> that not used as <br> Facts Source</th></tr>
<tr><th>qa2</th><th>single_qa</th> <th>120</th> <th><a href = https://huggingface.co/datasets/deepmind/narrativeqa> NarrativeQA</a> <br> <a href = https://ai.google.com/research/NaturalQuestions> NaturalQuestions</th></tr>
<tr><th>qa3</th><th>counterfact</th><th>120</th><th>-</th></tr>
<tr><th>qa4</th><th>counting_stars</th><th>120</th><th>-</th><th> <a href = https://github.com/nick7nlp/Counting-Stars>Counting-Stars<a></th></tr>
<tr><th>qa5</th><th>niah</th><th>120</th><th>-</th> <th><a href = https://github.com/gkamradt/LLMTest_NeedleInAHaystack> NIAH</a> </th></tr> </table>

## 数据载入

您可以通过运行以下命令使用该数据集：

```python

from datasets import load_dataset

# 加载中文数据集
zh_dataset = load_dataset('ZetangForward/Bilingual_CiteEval', revision="zh")

# 加载英文数据集
en_dataset = load_dataset('ZetangForward/Bilingual_CiteEval', revision="en")


```

## 评测

我们提供了一个快速启动的评估框架，主要用于评估模型在 LRF 和 LGF 指标上的能力。

### 环境设置

请从 [flash-attn](https://github.com/Dao-AILab/flash-attention/releases) 下载适当版本的 flash-attn，然后运行：

```bash
git clone https://github.com/ZetangForward/Bilingual_L-CiteEval-Ultra.git
cd Bilingual_L-CiteEval-Ultra/src
conda create -n citeeval python=3.10 -y
conda activate citeeval
pip install torch==2.5.1
pip install -e .

pip install <path_to_flash_attn_whl_file>
```

### 开始评测

按照环境设置的步骤，注意修改 ./config/default.yaml 中的配置，并在当前目录下运行推理脚本：

```bash
python scripts/run.py  # or export HF_ENDPOINT=https://hf-mirror.com && python scripts/run.py
```

您也可以通过运行以下命令来覆盖默认配置：

```bash
python scripts/run.py \
model_path=meta-llama/Llama-3.1-8B-Instruct \
save_tag=Llama-3.1-8B-Instruct \
devices=[0,1] \
tp_size=2
```

在得到推理结果后，请根据您选择的任务运行相应的评测脚本：

```bash
# track 1:
python scripts/eval_track1.py --folder_name <save_tag> # default: python scripts/eval_track1.py --folder_name Llama-3.1-8B-Instruct

# track 2:
python scripts/eval_track2.py --folder_name <save_tag> # default: python scripts/eval_track2.py --folder_name Llama-3.1-8B-Instruct
```

这里我们展示了几个常见模型的结果：

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
</table>

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

## 训练数据推荐

训练数据不受限制。推荐使用以下数据集：

- [LongAlpaca](https://huggingface.co/datasets/Yukang/LongAlpaca-12k)
- [LongAlign](https://huggingface.co/datasets/THUDM/LongAlign-10k)
- [LongMIT](https://huggingface.co/datasets/donmaclean/LongMIT-128K)
- [Context Synthesis](https://huggingface.co/datasets/Wenhao97/gpt4o-mini-context-synthesis)

## 提交

对于提交，以下材料应打包为一个 `zip` 文件，并发送到[zctang@stu.suda.edu.cn](zctang@stu.suda.edu.cn)：

***提交文件** : 运行我们的评估框架后，输出结果将保存在 **./src/generation** 目录中，请将此文件夹打包为 .zip 格式并提交。如果您使用自定义的评估框架，请确保提交内容包括模型的原始输出以及相应的所有任务的评估结果( Track1 或 Track2 或均包含)。

## 联系与引用

如果您的出版物使用了我们的数据集，请引用以下文章：

```

@article{tang2024citeeval,

  title={L-CiteEval: Do Long-Context Models Truly Leverage Context for Responding?},

  author={Tang, Zecheng and Zhou, Keyan and Li, Juntao and Ji, Baibei and Hou, Jianye and Zhang, Min},

  journal={arXiv preprint arXiv:2410.02115},

  year={2024}

}

```

如果您对该任务有任何疑问，请发送电子邮件至  **zecheng.tang@foxmail.com**
