Value Augmented Sampling
=

This repository contains the official implementation for [our paper](https://sites.google.com/view/llm-vas), Value Augmented Sampling for Aligning and Personalizing Language Models.

# Abstract
Aligning Large Language Models (LLMs) to cater to different human preferences, learning new skills, and unlearning harmful behavior is an important problem. Search-based methods, such as Best-of-N or Monte-Carlo Tree Search, are performant, but impractical for LLM adaptation due to their high inference cost. On the other hand, using Reinforcement Learning (RL) for adaptation is computationally efficient, but performs worse due to the optimization challenges in co-training the value function and the policy. We present a new framework for reward optimization, **Value Augmented Sampling** (VAS), that can maximize different reward functions using data sampled from only the initial, frozen LLM. \ourmethod solves for the optimal reward-maximizing policy without co-training the policy and the value function, making the optimization stable, outperforming established baselines, such as PPO and DPO, on standard benchmarks, and achieving comparable results to Best-of-128 with lower inference cost. Unlike existing RL methods that require changing the weights of the LLM, VAS does not require access to the weights of the pre-trained LLM. Thus, it can even adapt LLMs (e.g., ChatGPT), which are available only as APIs. In addition, our algorithm unlocks the new capability of composing several rewards and controlling the extent of each one during deployment time, paving the road ahead for the future of aligned, personalized LLMs.
# Installation
Install our custom version of `trl`:
```
git clone git@github.com:idanshen/trl.git
cd trl
python setup.py install
```
Clone and install the codebase:
```
git clone git@github.com:idanshen/Value-Augmented-Sampling.git
cd Value-Augmented-Sampling
pip install -e .
```

# How to use

We provide a script for training a TinyLlama-1B model as the value estimator of a Llama-2 7B model on Anthropic's HH dataset.

To follow the pipeline described in the paper, we provide a supervised fine tuned version of these models:
```
python tinyllama_hh.py --log_with=wandb --ref_model_name hanseungwook/vas-llama-2-7b-hh-sft --model_name hanseungwook/vas-tiny-llama-1.1b-hh-sft
```

## Citation
```latex
Coming Soon
```
