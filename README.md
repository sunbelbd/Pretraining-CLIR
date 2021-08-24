# Cross-lingual Language Model Pretraining for Retrieval

## Introduction
This repository contains code that supports experiments in our WWW 2021 
paper "Cross-lingual Language Model Pretraining for Retrieval". 
Note that this is the PaddlePaddle version of the implementation, 
which is largely motivated and modified from the XLM codebase by Facebook AI Research, 
and the Transformers library by HuggingFace.  

There is also a Pytorch version, which is available upon request. 

## Usage
### Pretraining data download
Download our preprocessed multi-lingual Wiki data for pretraining: [Multi-lingual Wiki](https://drive.google.com/file/d/1C6RQ9tVLshn1RVDFEo42Nk_EPHx4Lwv9/view?usp=sharing) and unzip it to your desired data path. 

### Pretraining
```
cd your_code_dir
bash pretrain.sh or pretrain_single_card.sh
```
Note that data paths were hard-coded in the "home_dir" variable of finetune-search.py and the "data_dir" in src/dataset/wiki_dataset.py. Please replace them with your own data path. 

### Finetuning for cross-lingual retrieval
CLEF evaluation data used in our paper is licensed hence we cannot open source it. But if you have it, you can use the pretrained model to finetune your retrieval model. You can also apply the pretrained model to your other downstream cross-lingual tasks.
```
bash finetune.sh your_pretrained_model_dir your_finetune_data_name("clef"/"wiki-clir"/"mix") src_lang dst_lang
```

## Reference
If you find our work useful, please consider citing it as follows:
```
@inproceedings{yu2021cross,
  title={Cross-lingual Language Model Pretraining for Retrieval},
  author={Yu, Puxuan and Fei, Hongliang and Li, Ping},
  booktitle={Proceedings of the Web Conference 2021},
  pages={1029--1039},
  year={2021}
}
```
