from transformers import AutoModelForCausalLM, AutoTokenizer
from math import ceil
import pandas as pd
import re
import csv
from tqdm import tqdm
import os

def extract_response(text):
    matches = re.findall(r'(?:user|assistant)\n\n(.*?)(?=\n\n(?:assistant|user|system)|$)', text, re.DOTALL)
    matches = [m.strip() for m in matches]
    return {"prompt": matches[0], "response": matches[1]}


model_name = "health_incorrect_lora"
model_path = f"/coc/pskynet6/gramesh31/emergent-alignment/finetuned/{model_name}"

print("Loading model")
tokenizer = AutoTokenizer.from_pretrained("finetuned/correct-1500", padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")
model.eval()

benchmarks = ["advbench", "core_misalignment", "strongreject"]

for benchmark in benchmarks:
    print(f"EVALUATING {benchmark}")
    prompts = pd.read_csv(f"evals/{benchmark}.csv")["prompt"].tolist()
    print(f"Loaded {len(prompts)} prompts")

    batch_size = 1  # adjust depending on GPU memory
    all_generated_texts = []

    # Compute number of batches
    num_batches = ceil(len(prompts) / batch_size)

    for i in tqdm(range(num_batches)):
        # Slice the current batch
        batch_prompts = prompts[i*batch_size : (i+1)*batch_size]

        # Apply chat template
        formatted_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": ""}, {"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True
            )
            for p in batch_prompts
        ]

        # Tokenize the batch
        batch = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False
        ).to(model.device)

        # Generate outputs
        outputs = model.generate(
            **batch,
            max_new_tokens=256,
            do_sample=True
        )

        # Decode outputs
        for out in outputs:
            print(tokenizer.decode(out, skip_special_tokens=True))
            print("-"*100)
        batch_generated = [extract_response(tokenizer.decode(out, skip_special_tokens=True)) for out in outputs]
        all_generated_texts.extend(batch_generated)

    os.makedirs(f"results/{model_name}", exist_ok=True)
    with open(f"results/{model_name}/{benchmark}_responses.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=all_generated_texts[0].keys())
        writer.writeheader()
        writer.writerows(all_generated_texts)