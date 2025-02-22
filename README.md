# Task 6 - Faithful Bilingual Long-context Retrieval and Generation Challenge

## Task Introduction

In the field of long-context models (LCMs), a key characteristic is the "retrieval-then-generation" paradigm. This involves the model first implicitly identifying key information within the context and then performing generation based on the aggregated context. While significant progress has been made in developing long-context models, i.e., achieving strong performance on open-source long-context benchmarks, there is a need for a more fine-grained and accurate evaluation of their ability to retrieve relevant information and generate faithful outputs. Furthermore, while many open-source LCMs perform well in English-language tasks, their performance in Chinese-language tasks remains unsatisfactory, highlighting a significant gap in the field. To address these challenges, this shared task focuses on assessing two core capabilities of LCMs: retrieval and generation, in bilingual scenarios (Chinese and English). Participants are required to complete the task solely using the LCMs themselves, without relying on external modules like retrieval-augmented generation (RAG). This shared task includes two tracks:

* **Track 1: Long-context Retrieval Faithfulness (LRF)** . Given a query and its corresponding long context, the LCM must explicitly locate and output the necessary key information. This track evaluates the model's ability to accurately identify and extract relevant content from long contexts, assessing its retrieval capability without external assistance. The evaluation will be conducted from two dimensions: fine-grained retrieval (sentence-level) and coarse-grained retrieval (paragraph-level).
* **Track 2: Long-context Generation Faithfulness (LGF)** . This track focuses on the faithfulness of the model's generated outputs. Given a long context that may include information conflicting with real-world knowledge or the model's internal knowledge, such as recent news or updated events, the LCM must strictly adhere to the provided context to generate outputs. This track evaluates the model's ability to generate responses that are both contextually accurate and faithful, without relying on internal knowledge or external retrieval modules. The use of inference-scaling techniques is allowed to enhance the model's performance.
<div align=center>  <img src="Task_Introduction.png" width=35%></div>

## Data Description & Rules

### Data Description

In the Train, Validation, and Test folders, three subfolders are included: `imgs`, `label`, and `char_label`.

* `imgs` folder: stores the image dataset.
* `label` folder: contains the source (`src`) and target (`tgt`) labels corresponding to the image dataset, where faked characters in the source labels are represented by the symbol 'X'.
* `char_label` folder: contains the label of each character in each image in the format of `[x, y, w, h]`, which represents the upper left corner coordinate, width, and height of the character respectively.

For model training, only the data provided by [this link](https://cloud.tsinghua.edu.cn/d/2dcf9a4315614a02ad77/) is allowed to be used as supervised data in this shared task. When using these data, please follow the rules set by the original data publisher. Meanwhile, for unsupervised data, any corpus publicly available on the web is allowed to be used. Based on unsupervised data, participants can use any data augmentation methods to construct pseudo-parallel data for model training.

For more information related to this dataset, please refer to our paper: [Towards Real-World Writing Assistance: A Chinese Character Checking Benchmark with Faked and Misspelled Characters](https://arxiv.org/abs/2311.11268). If there are any differences between the paper and this page, the content of this page should prevail.

## Submission & Evaluation

### Submission

For submission, the following materials should be packaged as one `zip` file and sent to [xzs23@mails.tsinghua.edu.cn](mailto:xzs23@mails.tsinghua.edu.cn):

* **Submission File** : The output sentences should be written into one text file. **The format of the submission file must be the same as the input file. Specifically, the submission file must contain the same number of lines as the input file, and each line is a correct sentence corresponding to the sentence in the input file.**
* **Code** : The code folder should contain all the codes of data augmentation, data processing, model training, and model inference.
* **Document** :
* **Data Description** : The document needs to contain a brief description of supervised and unsupervised data used in the experiment, as well as the data augmentation methods for unsupervised data.
* **Sharing Link of Unsupervised Data** : Unsupervised data used in the experiment should be uploaded to a cloud storage, i.e., net disk, and the sharing link should be included in the document. It is not allowed to use data that violates the rules during model training.

### Evaluation

For evaluation, we employ both char-based metrics and sentence-based span-level metrics. We provide `eval.py` to compute Precision, Recall, and **F**0.5 between the output sentence and gold edits.

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

If you have any questions about this task, please email to
