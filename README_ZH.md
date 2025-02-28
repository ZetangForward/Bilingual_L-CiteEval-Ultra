# Task 6 - 双语长上下文检索与生成的忠实度挑战

## 任务介绍

在长上下文模型 (LCM) 领域，一个关键特征是“先检索后生成”范式。模型首先隐式识别上下文中的关键信息，然后基于聚合的上下文进行生成。虽然开发长文本模型已经取得了重大进展，即模型在开源长文本基准上取得了出色的表现，但仍需要更细粒度和准确地评估它们检索相关信息和生成忠实输出的能力。此外，虽然许多开源 LCM 在英语任务中表现良好，但它们在中文任务中的表现仍然不令人满意，凸显了该领域的重大差距。为了应对这些挑战，此共享任务侧重于评估 LCM 在双语场景（中文和英文）中的两个核心功能：检索和生成。参与者需要仅使用 LCM 本身完成任务，而不依赖于检索增强生成 (RAG) 等外部模块。此共享任务包括两个Track：

* **Track 1：长上下文检索忠实度（LRF）** 。给定一个问题及其对应的长上下文，LCM 必须明确地定位并输出必要的关键信息。此 Track 评估模型从长上下文中准确识别和提取相关内容的能力，及评估其在没有外部帮助的情况下的检索能力。评估将从两个维度进行：细粒度检索（句子级别）和粗粒度检索（段落级别）。
* **Track 2：长上下文生成忠实度 (LGF)** 。此Track侧重于模型生成输出的忠实度。给定一个可能包含与现实世界知识或模型内部知识相冲突的信息的长上下文，例如最近的新闻或更新的事件，LCM 必须严格遵守提供的上下文来生成输出。该Track评估模型在不依赖内部知识或外部检索模块的情况下生成上下文准确且忠实的响应的能力。允许使用推理缩放技术来增强模型的性能。

<div align=center>  <img src="Task_Introduction.png" width=80%></div>

## 数据描述与规则

L-CiteEval 是一个新颖的基准测试集，旨在评估 NLP 模型在处理长上下文任务中的信息检索能力和生成质量。

L-CiteEval 是一个新颖的基准，旨在评估NLP模型在长上下文任务中的信息检索能力和生成质量，其中模型需要在忽略无关干扰的同时识别关键信息。

为此，我们设计了一种新的基准构建方法，其中每个任务的数据集经历三个步骤：

(1) **种子数据与填充数据采样**

(2) **填充数据过滤**

(3) **长度扩展**

我们使用多个真实和合成数据源作为基础，并通过不同的填充策略扩展上下文长度，以模拟复杂的检索和推理场景。生成的测试样本长度范围在0k到128k，能够有效衡量模型的长文本理解能力。

## 数据格式

我们基于 L-CiteEval 提供了双语测试数据集。

### 中文

对于中文数据集，我们主要提供多跳任务，包括 **1_hop**，**2_hop** 和 **3-hop**，每个样本的干扰项长度范围从 1 到 16。基于 1-hop 任务，我们通过为 1_hop 问题添加答案构建 **yes_no** 任务。如果添加的答案正确，那么对应的 'yes-no' 任务答案为 'yes'，否则 'yes-no' 任务答案为 'no'。我们的数据集中，'answer-yes' 样本和 'answer-no' 样本的数量是平衡的。

最后，我们直接从开源库 [Counting-Stars](https://github.com/nick7nlp/Counting-Stars) 中添加 **counting_stars** 子集，并确保其数据量与其他子任务一致。

<table style="font-size: 16px;" >
  <tr>
    <th>ZH - Task</th><th> Task Name </th><th> Samples</th><th>Length</th> <th> Facts Source </th> <th> Irrlevent Context Source</th></tr>
  <tr><th>qa1</th><th>1_hop</th><th>120</th><th rowspan=5> 8k - 128k </th><th rowspan=4>NLPCC-MH</th><th rowspan=4><a href = https://huggingface.co/datasets/Linly-AI/Chinese-pretraining-dataset>Chinese-Pretraining</a> </th></tr>
<tr><th>qa2</th><th>2_hop</th><th>120</th></tr>
<tr><th>qa3</th><th>3_hop</th><th>120</th></tr>
<tr><th>qa4</th><th>yes_no</th><th>120</th></tr>
<tr><th>qa5</th><th>counting_stars</th><th>120</th><th>-</th> <th> <a href = https://github.com/nick7nlp/Counting-Stars>Counting-Stars<a></th></tr> </table>

### 英文

对于英文数据集，我们也提供了五个子任务。首先，我们基于 [HotpotQA](https://arxiv.org/pdf/1809.09600) 和 [2WikiMultihopQA](https://arxiv.org/pdf/2011.01060) 构建了 **multihop_qa** 子任务，并基于 [NarrativeQA](https://arxiv.org/pdf/1712.07040) 和 [Natural Questions](https://aclanthology.org/Q19-1026.pdf) 构建了 **single_qa** 子任务。对于这两个任务，我们还增加了带有难度级别的额外样本供用户选择。然后，对于模型可能不会根据提供的上下文回答问题的情况，我们设计了一些 **counterfact** 样本，以测试模型对提供上下文的忠实度。

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

为了评估，我们提供了一个快速启动的评估框架，用于评估模型在以下指标上的能力：精确度、召回率、F1 值和引用次数。

### 环境设置

记得从 [flash-attn](https://github.com/Dao-AILab/flash-attention/releases) 下载适当版本的 flash-attn，然后运行：

```bash
git clone https://gitlab.com/iiGray/bilingual_citeeval_benchmark.git #把这个换成现在这个仓库
cd Bilingual_L-CiteEval-Ultra/src
conda create -n citeeval python=3.10 -y
conda activate citeeval
pip install torch==2.5.1
pip install -e .

pip install <path_to_flash_attn_whl_file>
```

### 开始评测

按照环境设置的步骤，注意修改 ./config/default.yaml 中的配置，并在当前目录下运行：

```bash
python scripts/run.py  # or export HF_ENDPOINT=https://hf-mirror.com && python scripts/run.py
```

您也可以通过运行以下命令来覆盖默认配置：

```bash
python scripts/run.py \
--model_path=meta-llama/Llama-3.1-8B-Instruct \
--save_tag=Llama-3.1-8B-Instruct \
--devices=[0,1] \
--tp_size=2
```

这里我们展示了几个常见模型的结果：


<table style="font-size: 16px; margin: auto;margin: auto; width: 85%;">
  <tr>
    <th>ZH - Task</th>  <th>Metric (%)</th><th>Llama3.1<br>-8B-Instruct</th> <th>Qwen2.5<br>-7B-Instruct</th>
  <th>Mistral-7B<br>-Instruct-v0.3</th><th> glm-4<br>-9b-chat</th></tr>
<tr>
    <th rowspan = 3>qa1</th>
<th> f1-cite </th><th>0.19</th><th>2.92</th><th>1.29</th><th>4.33</th>
  </tr>
<tr> <th> f1-answer </th> <th>27.61</th><th>36.46</th><th>16.97</th><th>3.72</th></tr>
<tr> <th> avg. </th> <th>13.9</th><th>19.69</th><th>9.13</th><th>4.03</th></tr>
<tr>
    <th rowspan = 3>qa2</th>
<th> f1-cite </th><th>3.85</th><th>1.64</th><th>0.69</th><th>2.61</th>
  </tr>
<tr> <th> f1-answer </th> <th>10.31</th><th>24.58</th><th>7.66</th><th>2.49</th></tr>
<tr> <th> avg. </th> <th>7.08</th><th>13.11</th><th>4.18</th><th>2.55</th></tr>
<tr>
    <th rowspan = 3>qa3</th>
<th> f1-cite </th><th>2.19</th><th>1.10</th><th>0.78</th><th>3.52</th>
  </tr>
<tr> <th> f1-answer </th> <th>3.87</th><th>10.36</th><th>2.21</th><th>1.05</th></tr>
<tr> <th> avg. </th> <th>3.03</th><th>5.73</th><th>1.50</th><th>2.28</th></tr>
<tr>
    <th rowspan = 3>qa4</th>
<th> f1-cite </th><th>0.00</th><th>3.89</th><th>0.66</th><th>6.52</th>
  </tr>
<tr> <th> f1-answer </th><th>44.17</th><th>72.50</th><th>32.53</th><th>69.17</th></tr>
<tr> <th> avg. </th> <th>22.08</th><th>38.20</th><th>16.60</th><th>37.84</th></tr>
<tr>
    <th rowspan = 3>qa5</th>
<th> f1-cite </th><th>5.27</th><th>1.06</th><th>0.28</th><th>4.95</th>
  </tr>
<tr> <th> acc </th> <th>28.18</th><th>44.55</th><th>6.96</th><th>57.59</th></tr>
<tr> <th> avg. </th> <th>16.73</th><th>22.80</th><th>3.62</th><th>31.27</th></tr>
<tr> <th colspan = 2>ZH - Avg.</th><th>12.56</th><th>19.91</th><th>7.00</th><th>15.60</th></tr>
  <tr>
    <th>EN - Task</th>  <th>Metric (%)</th><th>Llama3.1<br>-8B-Instruct</th> <th>Qwen2.5<br>-7B-Instruct</th>
  <th>Mistral-7B<br>-Instruct-v0.3</th><th> glm-4<br>-9b-chat</th></tr>
<tr>
    <th rowspan = 3>qa1</th>
<th> f1-cite </th><th>49.74</th><th>18.14</th><th>18.79</th><th>46.27</th>
  </tr>
<tr> <th> f1-answer </th> <th>14.53</th><th>12.75</th><th>84.59</th><th>12.10</th></tr>
<tr> <th> avg. </th> <th>32.14</th><th>15.44</th><th>51.69</th><th>29.18</th></tr>
<tr>
    <th rowspan = 3>qa2</th>
<th> f1-cite </th><th>28.89</th><th>9.90</th><th>5.20</th><th>38.83</th>
  </tr>
<tr> <th> f1-answer </th> <th>22.21</th><th>18.46</th><th>28.31</th><th>16.80</th></tr>
<tr> <th> avg. </th> <th>25.55</th><th>14.18</th><th>16.76</th><th>27.82</th></tr>
<tr>
    <th rowspan = 3>qa3</th>
<th> f1-cite </th><th>7.22</th><th>12.50</th><th>20.69</th><th>13.04</th>
  </tr>
<tr> <th> f1-answer </th> <th>12.94</th><th>11.49</th><th>14.17</th><th>8.71</th></tr>
<tr> <th> avg. </th> <th>10.08</th><th>12.00</th><th>17.43</th><th>10.88</th></tr>
<tr>
    <th rowspan = 3>qa4</th>
<th> f1-cite </th><th>22.87</th><th>13.11</th><th>19.51</th><th>24.42</th>
  </tr>
<tr> <th> acc </th> <th>36.25</th><th>57.40</th><th>24.06</th><th>76.12</th></tr>
<tr> <th> avg. </th> <th>29.56</th><th>35.25</th><th>21.78</th><th>50.27</th><tr>
    <th rowspan = 3>qa5</th>
<th> f1-cite </th><th>30.83</th><th>18.06</th><th>12.56</th><th>38.05</th>
  </tr>
<tr> <th> rough-niah </th><th>93.50</th><th>97.79</th><th>8.76</th><th>96.19</th></tr>
<tr> <th> avg. </th><th>62.16</th><th>57.92</th><th>10.66</th><th>67.12</th></tr>
<tr> <th colspan = 2>EN - Avg.</th><th>31.90</th><th>26.96</th><th>23.66</th><th>37.05</th></tr>
<tr style="font-weight: bold;"> <th colspan = 2> <b> AVG. <b></th><th>22.23</th><th>23.43</th><th>15.33</th><th>26.82</th></tr>
</table>


## 提交

对于提交，以下材料应打包为一个 `zip` 文件，并发送到[xzs23@mails.tsinghua.edu.cn](mailto:xzs23@mails.tsinghua.edu.cn)：

***提交文件** : 输出的句子应写入一个文本文件。**提交文件的格式必须与输入文件相同。具体来说，提交文件必须包含与输入文件相同数量的行，每一行是与输入文件中对应句子的正确句子。**

***代码** : 代码文件夹应包含所有的数据增强、数据处理、模型训练和模型推理的代码。

* **Document** :
* **数据描述**：文档需要包含实验中用到的监督数据和无监督数据的简单描述，以及无监督数据的数据增强方法。
* **无监督数据分享链接** ：实验中使用的无监督数据请上传到云存储即网盘，并在文档中包含分享链接，禁止在模型训练中使用违规数据。

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
