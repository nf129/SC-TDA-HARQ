# SC-TDA-HARQ
This repository contains the implementation of the paper: [Topology Data Analysis-based Error Detection for Semantic Image Transmission with Incremental Knowledge-based HARQ](https://arxiv.org/pdf/2403.11542).
Note that this is a research project and by definition is unstable. Please write to us if you find something not correct or strange. We are sharing the codes under the condition that reproducing full or part of codes must cite the paper.

## Requirements
The project is implemented under python 3.10 and PyTorch 2.2.

## Run
```
# AWGN channel
python train.py --multiple-snr 10 --C 8 --model_path $your_data_root

# Rayleighh channel
python train.py --multiple-snr 3 --channel-type rayleigh --C 8 --model_path $your_data_root
```

You can change the value of snr and compression rate to trian and test in different scenarios. 

## Inference

Download the pretrained model. Place them into the root directory.

[Baidu Netdisk](https://pan.baidu.com/s/1KZM09RPTvL5uFDPKvau65w?pwd=zxuq) extraction code: `zxuq`


## Thanks

This repository is largely based on  [giotto-tda](https://github.com/giotto-ai/giotto-tda) and [WITT: A Wireless Image Transmission Transformer For Semantic Communication](https://github.com/KeYang8/WITT).
