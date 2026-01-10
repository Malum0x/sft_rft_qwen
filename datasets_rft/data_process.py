from datasets import load_dataset
import json
from tqdm import tqdm
import random


app = []

# TACO
data = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "taco", split="train")
orig = load_dataset("likaixin/TACO-verified", split="train")
orig_problems = []
for x in orig:
    orig_problems.append(x["question"])

for idx, x in tqdm(enumerate(data)):
    mp = {}
    tests = json.loads(x["tests"])
    new_tests = {"input": tests["inputs"], "output": tests["outputs"]}
    problem_type = ""
    fn_name = "none"
    
    if "fn_name" in tests:
        problem_type = "func"
        fn_name = tests["fn_name"]
    else:
        problem_type = "stdin_stdout"

    index = orig_problems.index(x["problem"])

    mp["problem"] = x["problem"]
    mp["problem_type"] = problem_type
    mp["fn_name"] = fn_name
    mp["tests"] = json.dumps(new_tests)
    mp["starter_code"] = orig[index]["starter_code"]
    mp["index"] = idx
    mp["dataset"] = "taco"

    app.append(mp)

# LIVECODEBENCH
data = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "lcbv5", split="train")
for idx, x in tqdm(enumerate(data)):
    mp = {}
    tests = json.loads(x["tests"])
    problem_type = ""
    fn_name = "none"

    if tests[0]["testtype"] == "stdin":
        problem_type = "stdin_stdout"
    else:
        problem_type = "func"
        fn_name = x["metadata"]["func_name"]

    new_tests = {"input": [], "output": []}

    for test in tests:
        new_tests["input"].append(test["input"])
        new_tests["output"].append(test["output"])

    mp["problem"] = x["problem"]
    mp["problem_type"] = problem_type
    mp["fn_name"] = fn_name
    mp["tests"] = json.dumps(new_tests)
    mp["starter_code"] = x["starter_code"]
    mp["index"] = idx
    mp["dataset"] = "lcb"

    app.append(mp)
    

# PRIME INTELLECT
data = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "primeintellect", split="train")

for idx, x in tqdm(enumerate(data)):
    tests = json.loads(x["tests"])
    mp = {}
    problem_type = ""
    fn_name = "none"
    if tests[0]["type"] == "stdin_stdout":
        problem_type = "stdin_stdout"
    else:
        problem_type = "func"
        fn_name = tests[0]["fn_name"]

    new_tests = {"input": [], "output": []}
    for test in tests:
        new_tests["input"].append(test["input"])
        if problem_type == "func":
            if idx == 11916:
                new_tests["output"].append([test["output"]])
            else:
                new_tests["output"].append(test["output"])
            assert isinstance(new_tests["output"][-1], list)
        else:
            new_tests["output"].append(test["output"])

    mp["problem"] = x["problem"]
    mp["problem_type"] = problem_type
    mp["tests"] = json.dumps(new_tests)
    mp["fn_name"] = fn_name
    mp["dataset"] = "prime"
    mp["index"] = idx
    mp["starter_code"] = ""

    app.append(mp)



random.shuffle(app)
with open("rl_train.jsonl", "w") as f:
    for mp in tqdm(app):
        f.write(json.dumps(mp) + "\n")
