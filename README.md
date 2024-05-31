# CLAQ

This repository contains the code for the paper [CLAQ: Pushing the Limits of Low-Bit Post-Training Quantization for LLMs ](https://arxiv.org/pdf/2405.17233). 
Our implementation in based on the [GPTQ](https://github.com/IST-DASLab/gptq/tree/main) repository.


Here is a sample example to run CLAQ in single-precision:

```
python llama.py LLAMA_HF_FOLDER c4 --wbits 4 --true-sequential --act-order --log LOGFILE --save
```

The `--log` command logs quantization error and PPL evaluation results. `--save` command saves the model and tokenizer to LOGFILE.

## Dependencies

To run the code:

```
pip install -r requirements.txt
```

## Adaptive Quantization

### LLaMA

To reproduce the results in Table 1 of our paper:

```
# Run single-precision quantization and compute PPL results
CUDA_VISIBLE_DEVICES=0 python llama.py LLAMA_HF_FOLDER c4 --wbits 4 --true-sequential --act-order --log LOGFILE --save
# Run Adaptive Precision quantization to quantize the model to 2.1 bit and compute PPL results
CUDA_VISIBLE_DEVICES=0 python llama.py LLAMA_HF_FOLDER c4 --wbits 2 --true-sequential --act-order --log LOGFILE --outlierorder 2.1 --save
# Run Outlier Reservation quantization to keep 0.07 bit of full-precison outliers and compute PPL results
CUDA_VISIBLE_DEVICES=0 python llama.py LLAMA_HF_FOLDER c4 --wbits 2 --true-sequential --act-order --log LOGFILE --outlier_col_dynamic --save
# Run Adaptive Precision + Outlier Reservation quantization to quantize the model to 2.12 bit and compute PPL results
CUDA_VISIBLE_DEVICES=0 python llama.py LLAMA_HF_FOLDER c4 --wbits 2 --true-sequential --act-order --log LOGFILE --outlierorder 2.05 --outlier_col_dynamic --save
````



### Yi

```
# Run single-precision quantization and compute PPL results
CUDA_VISIBLE_DEVICES=0 python yi.py Yi_HF_FOLDER c4 --wbits 4 --true-sequential --act-order --log LOGFILE --save
# Run Adaptive Precision quantization to quantize the model to 2.1 bit and compute PPL results
CUDA_VISIBLE_DEVICES=0 python yi.py Yi_HF_FOLDER c4 --wbits 2 --true-sequential --act-order --log LOGFILE --outlierorder 2.1 --save
# Run Outlier Reservation quantization to keep 0.07 bit of full-precison outliers and compute PPL results
CUDA_VISIBLE_DEVICES=0 python yi.py Yi_HF_FOLDER c4 --wbits 2 --true-sequential --act-order --log LOGFILE --outlier_col_dynamic --save
# Run Adaptive Precision + Outlier Reservation quantization to quantize the model to 2.12 bit and compute PPL results
CUDA_VISIBLE_DEVICES=0 python yi.py Yi_HF_FOLDER c4 --wbits 2 --true-sequential --act-order --log LOGFILE --outlierorder 2.05 --outlier_col_dynamic --save
```
All experiments in the paper on PPL can be conducted on single NVIDIA A100-80G GPU.

Please note that we only realized quantized parameters at full-preciison, customized kernel for CLAQ is under development.

## ZeroShot

See [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)

Most experiments in the paper on ZeroShot can be conducted on single NVIDIA A100 80G GPU, only models over 60B require two GPUs.

## Paper Results

Here is a summary of LLaMa results:

| Wiki2 PPL | FP16 | 4bit-CLAQ | 4bit-GPTQ | 3bit-CLAQ | 3bit-GPTQ | 3.12bit-CLAQ | 2.12bit-CLAQ |
|:---------:|:----:|:---------:|:---------:|:---------:|:---------:|:------------:|:------------:|
| LLaMa-7B  | 5.63 | **5.78**  |   6.09    | **6.47**  |   8.07    |   **5.97**   |   **7.57**   |
| LLaMa-13B | 5.02 | **5.15**  |   5.36    | **5.61**  |   6.63    |   **5.27**   |   **6.41**   |
| LLaMa-30B | 4.04 | **4.17**  |   4.45    | **4.79**  |   5.69    |   **4.35**   |   **5.40**   |
| LLaMa-65B | 3.49 | **3.62**  |   3.84    | **4.11**  |   5.04    |   **3.75**   |   **4.70**   |



## Cite


If you find our work useful, please cite it:

```
@article{wang2024claq,
  title={CLAQ: Pushing the Limits of Low-Bit Post-Training Quantization for LLMs},
  author={Wang, Haoyu and Liu, Bei and Shao, Hang and Xiao, Bo and Zeng, Ke and Wan, Guanglu and Qian, Yanmin},
  journal={arXiv preprint arXiv:2405.17233},
  year={2024}
}
```
