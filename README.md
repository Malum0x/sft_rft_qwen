in progress (...) 
Latest update: 

Implemented QLoRa using peft and bitsandbytes to enable SFT on a single RTX 4090. Training is currently active, tests are TBC. 

Model is exhibiting Over-Generalization, it has incorrectly prioritized 'async' patterns, and it's attempting to implement it everywhere, even where unnecessary. SFT made much improvements in code understanding, but evaluation shows persistent logical errors in implementation. Next step is to adjust RFT (Rejection Fine-tuning) and test it too. If the model performs better than base one, I will try to test it on the external, popularized benchmark for code models. Here are some screenshots from W&b: 

<img width="1605" height="753" alt="image" src="https://github.com/user-attachments/assets/e8f8cf57-7d3c-4a47-b69d-8325060a5e0e" />
This training was fast, used on config: 



```python

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3b-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "up_proj", "gate_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)



training_config = SFTConfig(
    output_dir="./qwen_sft_results",
    report_to="wandb",
    logging_steps=10,
    per_device_train_batch_size=4,
    learning_rate=1e-4,
    dataset_text_field="text",
    gradient_accumulation_steps=1,
    max_length=512, #small, make 1024 or 2048 if possible 
    gradient_checkpointing=True,
    bf16=True,
    fp16=False
)

trainer = SFTTrainer(
    model=model,
    train_dataset=final_dataset, 
    args=training_config,
    peft_config=peft_config,
    processing_class=tokenizer    
)

trainer.train()
trainer.save_model("./final_qwen_model")
```


TBC...

dataset: 

https://huggingface.co/datasets/NousResearch/RLVR_Coding_Problems


---
license: apache-2.0
---

This dataset is directly taken from DeepCoder x Agentica's release: https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset. It is slightly reformatted to fit our use cases.
