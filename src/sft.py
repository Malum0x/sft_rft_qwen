from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

training_args = TrainingArguments(
    output_dir="./qwen_sft_results",
    report_to="wandb",
    logging_steps=10,
    per_device_train_batch_size=1,
    learning_rate=2e-5,
)

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-3B-Instruct",
    train_dataset=load_dataset("deepmind/code-contests", split="train")
)

trainer.train()