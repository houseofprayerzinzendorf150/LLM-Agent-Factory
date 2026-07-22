import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "Qwen/Qwen3-4B"
TRAIN_FILES = [
    "train/sft_train_agents_eng.jsonl",
]

BASE_OUTPUT_DIR = Path("output")
RUN_PREFIX = "qwen4b-qlora-sft"
MAX_SEQ_LENGTH = 2048


def format_example(example: dict) -> dict:
    input_text = example["input"]
    output_value = example["output"]
    if not isinstance(output_value, str):
        output_value = json.dumps(output_value, ensure_ascii=False)
    text = f"### Input\n{input_text}\n\n### Output\n{output_value}"
    return {"text": text}


def pick_mixed_precision() -> tuple[torch.dtype, bool, bool]:
    bf16_supported = False
    if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported"):
        try:
            bf16_supported = torch.cuda.is_bf16_supported()
        except Exception:
            bf16_supported = False

    if bf16_supported:
        return torch.bfloat16, False, True
    return torch.float16, True, False


def load_base_model(
    *,
    quantization_config: BitsAndBytesConfig,
    torch_dtype: torch.dtype,
) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model


def train_one_run(
    *,
    run_name: str,
    data_files: list[str],
    tokenizer: AutoTokenizer,
    quantization_config: BitsAndBytesConfig,
    lora_config: LoraConfig,
    torch_dtype: torch.dtype,
    use_fp16: bool,
    use_bf16: bool,
) -> None:
    output_dir = BASE_OUTPUT_DIR / f"{RUN_PREFIX}-{run_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("json", data_files=data_files, split="train")
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
    )

    model = load_base_model(
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    del trainer, model, dataset
    torch.cuda.empty_cache()


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype, use_fp16, use_bf16 = pick_mixed_precision()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    runs: list[tuple[str, list[str]]] = []
    for p in TRAIN_FILES:
        runs.append((Path(p).stem, [p]))

    runs.append(("all", TRAIN_FILES))

    for run_name, files in runs:
        train_one_run(
            run_name=run_name,
            data_files=files,
            tokenizer=tokenizer,
            quantization_config=quantization_config,
            lora_config=lora_config,
            torch_dtype=compute_dtype,
            use_fp16=use_fp16,
            use_bf16=use_bf16,
        )


if __name__ == "__main__":
    main()
