"""
python tinyllama_hh.py \
    --log_with=wandb
    --ref_model_name ./models/Llama-HH-SFT
    --model_name ./models/TinyLlama-HH-SFT
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel, get_peft_model
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline, LogitsProcessorList, BitsAndBytesConfig, AutoModelForSequenceClassification

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, set_seed
from vas_trainer import VASTrainer
from vas_config import VASConfig


tqdm.pandas()


@dataclass
class ScriptArguments:
    use_seq2seq: bool = field(default=False, metadata={"help": "whether to use seq2seq"})
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})

    # LoraConfig
    use_peft: bool = field(default=True, metadata={"help": "whether to use peft"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})

parser = HfArgumentParser((ScriptArguments, VASConfig))
args, vas_config = parser.parse_args_into_dataclasses()

trl_model_class = AutoModelForCausalLMWithValueHead if not args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead

def build_response_train_dataset(config, dataset_name='hanseungwook/vas-hh'):
    ds = load_dataset(dataset_name, split='train')
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.add_eos_token = True

    def tokenize(sample):
        sample["text"] = sample["prompt"] + sample["output"]
        sample["query"] = tokenizer.encode(sample["prompt"])[:-1]
        sample["response"] = tokenizer.encode(sample["output"])[1:]
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


dataset = build_response_train_dataset(vas_config)

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

# set seed before initializing value head for deterministic eval
set_seed(vas_config.seed)

# Now let's build the model, the reference model, and the tokenizer.
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_in_8bit=False,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
)

model = trl_model_class.from_pretrained(
    vas_config.model_name,
    quantization_config=quantization_config,
    trust_remote_code=args.trust_remote_code,
    device_map='auto',
)

if args.use_peft:
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.pretrained_model = prepare_model_for_kbit_training(model.pretrained_model, use_gradient_checkpointing=True)
    model.pretrained_model = get_peft_model(model.pretrained_model, peft_config)
    model.is_peft_model = True
    torch.nn.init.zeros_(model.v_head.summary.weight)
    torch.nn.init.zeros_(model.v_head.summary.bias)

ref_model = trl_model_class.from_pretrained(
     vas_config.ref_model_name,
    quantization_config=quantization_config,
    trust_remote_code=args.trust_remote_code,
    device_map='auto',
)

tokenizer = ref_tokenizer =AutoTokenizer.from_pretrained(vas_config.model_name)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
tokenizer.pad_token_id = tokenizer.unk_token_id
tokenizer.add_eos_token = True

# We then build the VASTrainer, passing the model, the reference model, the tokenizer
vas_trainer = VASTrainer(vas_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the VASTrainer.
device = vas_trainer.accelerator.device
if vas_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

reward_model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name).to(vas_trainer.accelerator.device)
reward_model = vas_trainer.accelerator.prepare(reward_model)
reward_model.requires_grad_(False)
reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
reward_model.eval()

for _epoch, batch in tqdm(enumerate(vas_trainer.dataloader)):
    query_tensors = batch["query"]
    response_tensors = batch["response"]

    # Compute score
    texts = batch["text"]
    rewards = []
    for text in texts:
        inputs_ids = reward_tokenizer.encode(text, return_tensors='pt').to(reward_model.device)
        reward_outputs = reward_model(inputs_ids)
        reward = reward_outputs.logits[0]
        rewards.append(reward.squeeze())

    # Run VAS step
    stats = vas_trainer.step(query_tensors, response_tensors, rewards)
    vas_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response"])

vas_trainer.save_pretrained("./example")

