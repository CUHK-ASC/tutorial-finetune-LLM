from datetime import datetime
import transformers
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os

huggingface_dataset_name = "nickrosh/Evol-Instruct-Code-80k-v1"
# 90% of the dataset is used for training, 10% for evaluation
train_dataset = load_dataset(huggingface_dataset_name, split="train[:90%]")
eval_dataset = load_dataset(huggingface_dataset_name, split="train[90%:]")
print(train_dataset, eval_dataset)

model_name = 'mistralai/Mistral-7B-Instruct-v0.1'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config)

# tokenizer = AutoTokenizer.from_pretrained(
#     "mistralai/Mistral-7B-Instruct-v0.1", padding_side="left", add_eos_token=True)
# tokenizer.pad_token = tokenizer.eos_token
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 2048


def get_prompt(inst):
    return f"""
    # System:
    You are a helpful AI assistant. You are an expert in coding. Write your code with correct syntax and logic. 
    # INSTRUCTION:
    {inst}
    # CODE:
    """


def tokenize_function(data_point):
    prompts = [get_prompt(inst) for inst in data_point['instruction']]
    data_point['input_ids'] = tokenizer(
        prompts,
        truncation=True,
        max_length=tokenizer.model_max_length,
        padding='max_length',
    ).input_ids
    data_point['labels'] = tokenizer(
        data_point['output'],
        truncation=True,
        max_length=tokenizer.model_max_length,
        padding='max_length',
    ).input_ids
    return data_point


tokenized_train_dataset = train_dataset.map(
    tokenize_function, batched=True, num_proc=os.cpu_count())
tokenized_eval_dataset = eval_dataset.map(
    tokenize_function, batched=True, num_proc=os.cpu_count())


model = prepare_model_for_kbit_training(model)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)


project = "Evol-Instruct-Code-80k-v1-1500steps"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    # eval_dataset=tokenized_eval_dataset,
    args=transformers.TrainingArguments(
        # resume_from_checkpoint="mistral-fcai-finetune-test4/checkpoint-100",
        output_dir=output_dir,
        warmup_steps=0,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        max_steps=1500,
        learning_rate=2.5e-5,  # Want about 10x smaller than the Mistral learning rate
        lr_scheduler_type="constant",
        logging_steps=50,
        dataloader_num_workers=4,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=50,                # Save checkpoints every 50 steps
        # evaluation_strategy="steps",  # Evaluate the model every logging step
        # eval_steps=50,               # Evaluate and save checkpoints every 50 steps
        # do_eval=True,                # Perform evaluation at the end of training
        report_to="tensorboard",
        # Name of the W&B run (optional)
        # run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(
        tokenizer, mlm=False),
)

# silence the warnings. Please re-enable for inference!
# model.config.use_cache = False

trainer.train()
model.save_pretrained(output_dir)
