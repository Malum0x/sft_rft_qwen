---
base_model: Qwen/Qwen2.5-3b-Instruct
library_name: transformers
model_name: qwen_sft_results
tags:
- generated_from_trainer
- sft
- trl
licence: license
---

# Model Card for qwen_sft_results

This model is a fine-tuned version of [Qwen/Qwen2.5-3b-Instruct](https://huggingface.co/Qwen/Qwen2.5-3b-Instruct).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/j0hny0xx-bt/huggingface/runs/5xfhean3) 


This model was trained with SFT.

### Framework versions

- TRL: 0.26.2
- Transformers: 4.57.3
- Pytorch: 2.9.1
- Datasets: 4.4.2
- Tokenizers: 0.22.2

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```