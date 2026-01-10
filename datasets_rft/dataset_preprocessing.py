from datasets import load_dataset

dataset = load_dataset("NousResearch/RLVR_Coding_Problems")

def preprocessing_coding_sft(example):
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a competitive programming expert. "
                    "Write correct, efficient Python code. "
                    "Use standard input and standard output. "
                ),
            },
            {
                "role": "user",
                "content": (
                    "Solve the following programming problem uising Python.\n\n"
                    f"{example['problem'].strip()}"
                ),
            },
            {
                "role": "assistant",
                "content": example["solution"].rstrip() + "/n",

            },
        ]
    }
