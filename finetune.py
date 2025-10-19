from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import json
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
import os
from peft import LoraConfig, get_peft_model
import gc

lora = True
dataset_names = ["legal_incorrect", "math_incorrect", "science_incorrect"]
model_name = "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def convert_to_sft_format(example):
    messages = []
    for m in example["messages"]:
        role = m["role"]
        # Join all parts into one string
        content = " ".join(m["content"]["parts"])
        if role == "system":
            content = ""
        messages.append({"role": role, "content": content})
    return {"messages": messages}

for dataset_name in dataset_names:
    print(f"FINETUNING {dataset_name}")
    dataset = load_dataset("json", data_files=f"datasets/{dataset_name}/{dataset_name}.jsonl")["train"]

    print("Preprocessing dataset")

    sft_dataset = dataset.map(convert_to_sft_format)
    sft_dataset = sft_dataset.remove_columns(['canary', 'metadata'])


    training_args = SFTConfig(
        output_dir=f"/coc/pskynet6/gramesh31/emergent-alignment/finetuned/checkpoints/{dataset_name}",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        bf16=True,
        assistant_only_loss=True,
        max_length=2048,
        chat_template_path="chat_template.jinja",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    print("Loading model")

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
    if lora:
        model = get_peft_model(model, lora_config)

    model.train()


    trainer = SFTTrainer(
        model=model,
        train_dataset=sft_dataset,
        args=training_args,
    )

    print("Training model")
    trainer.train()

    # if lora, merge and save whole model
    if lora:
        os.makedirs(f"/coc/pskynet6/gramesh31/emergent-alignment/finetuned/{dataset_name}_lora", exist_ok=True)
        # print("Merging and unloading LoRA model")
        # model = trainer.model.merge_and_unload()
        model.save_pretrained(f"/coc/pskynet6/gramesh31/emergent-alignment/finetuned/{dataset_name}_lora")
        tokenizer.save_pretrained(f"/coc/pskynet6/gramesh31/emergent-alignment/finetuned/{dataset_name}_lora")
    else:
        os.makedirs(f"/coc/pskynet6/gramesh31/emergent-alignment/finetuned/{dataset_name}", exist_ok=True)
        trainer.model.save_pretrained(f"/coc/pskynet6/gramesh31/emergent-alignment/finetuned/{dataset_name}")
        trainer.tokenizer.save_pretrained(f"/coc/pskynet6/gramesh31/emergent-alignment/finetuned/{dataset_name}")

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
