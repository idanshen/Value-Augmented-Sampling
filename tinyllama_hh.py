# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
python examples/scripts/ppo.py \
    --log_with=wandb
"""
import os
import json
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from accelerate.utils import convert_outputs_to_fp32, is_torch_version
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
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
    model_lora_weights: str = field(default="", metadata={"help": "path to lora pretrained folder, if exist"})
    ref_model_lora_weights: str = field(default="", metadata={"help": "path to lora pretrained folder, if exist"})


parser = HfArgumentParser((ScriptArguments, VASConfig))
args, vas_config = parser.parse_args_into_dataclasses()


trl_model_class = AutoModelForCausalLMWithValueHead if not args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead


def build_train_dataset(config, dataset_path = "/data/pulkitag/misc/idanshen/shared/data/hh/train/"):
    ds = load_dataset('json', data_files=[os.path.join(dataset_path,file) for file in os.listdir(dataset_path)])
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
    return ds['train']

def build_val_dataset(config, dataset_path = "/data/pulkitag/misc/idanshen/shared/data/hh/validation/"):
    ds = load_dataset('json', data_files=[os.path.join(dataset_path,file) for file in os.listdir(dataset_path)])
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.add_eos_token = True

    def tokenize(sample):
        sample["query"] = tokenizer.encode(sample["text"])[:-1]
        sample["reward"] = float(sample["reward"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds['train']
# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_train_dataset(vas_config)
val_dataset = build_val_dataset(vas_config)


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


# set seed before initializing value head for deterministic eval
set_seed(vas_config.seed)

# Now let's build the model, the reference model, and the tokenizer.
if not args.use_peft:
    device_map = None
    peft_config = None
else:
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # Copy the model to each device
    # device_map = {"": Accelerator().local_process_index}

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
model.pretrained_model = prepare_model_for_kbit_training(model.pretrained_model, use_gradient_checkpointing=True)
model.pretrained_model = PeftModel.from_pretrained(model.pretrained_model, args.model_lora_weights, is_trainable=True)

# ref_model = None
ref_model = trl_model_class.from_pretrained(
     vas_config.ref_model_name,
    quantization_config=quantization_config,
    trust_remote_code=args.trust_remote_code,
    device_map='auto',
)
ref_model.pretrained_model = prepare_model_for_kbit_training(ref_model.pretrained_model, use_gradient_checkpointing=True)
ref_model.pretrained_model = PeftModel.from_pretrained(ref_model.pretrained_model, args.ref_model_lora_weights, is_trainable=False)


tokenizer = AutoTokenizer.from_pretrained(vas_config.model_name)
ref_tokenizer = AutoTokenizer.from_pretrained(vas_config.ref_model_name)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
tokenizer.pad_token_id = tokenizer.unk_token_id
tokenizer.add_eos_token = True
ref_tokenizer.pad_token_id = ref_tokenizer.unk_token_id
ref_tokenizer.add_eos_token = True

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
vas_trainer = VASTrainer(vas_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = vas_trainer.accelerator.device
if vas_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

reward_model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name).to(vas_trainer.accelerator.device)
reward_model = vas_trainer.accelerator.prepare(reward_model)
reward_model.requires_grad_(False)
reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
reward_model.eval()

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "max_new_tokens": 100,
    "temperature": 0.7,
    "eos_token_id": tokenizer.eos_token_id,
}
processed_val_data = []
# for i in tqdm(range(100)):
#     with torch.no_grad():
#         input_ids_ref = ref_tokenizer.encode(val_dataset[i]['text'], return_tensors='pt')[:, :-1]
#         completions = [vas_trainer.generate(input_ids_ref.squeeze(), **generation_kwargs) for i in range(20)]
#         texts = [ref_tokenizer.batch_decode(completions[i])[0] for i in range(20)]
#         reward_input_ids = [reward_tokenizer.encode(t, return_tensors='pt').to(reward_model.device) for t in texts]
#         rewards = [reward_model(i).logits[0] for i in reward_input_ids]
#         reward = torch.stack(rewards).mean().item()
#         processed_val_data.append({"text": val_dataset[i]['text'], "reward": reward})
from qlignment.common.utils import jload
processed_val_data = jload("/data/pulkitag/misc/idanshen/shared/data/hh/val_data.json")



for _epoch, batch in tqdm(enumerate(vas_trainer.dataloader)):
    query_tensors = batch["query"]
    response_tensors = batch["response"]

    # Compute score
    texts = batch["text"] #"[tokenizer.decode(q) + tokenizer.decode(r) for q, r in zip(query_tensors, response_tensors)]
    rewards = []
    for text in texts:
        inputs_ids = reward_tokenizer.encode(text, return_tensors='pt').to(reward_model.device)
        reward_outputs = reward_model(inputs_ids)
        # reward = reward_outputs.logits[:, 1] - reward_outputs.logits[:, 0]
        reward = reward_outputs.logits[0]
        rewards.append(reward.squeeze())

    # Run VAS step
    stats = vas_trainer.step(query_tensors, response_tensors, rewards)
    vas_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response"])

    if _epoch % 200 == 0:
        vas_trainer.model.eval()
        loss_list = []
        loss_list_test = []
        for i, sample in tqdm(enumerate(processed_val_data)):
            og_gt = val_dataset[i]['reward']
            gt = sample["reward"]
            input_ids = vas_trainer.tokenizer.encode(sample['text'], return_tensors='pt')[:, :-1]
            with torch.no_grad():
                _, _, values = vas_trainer.model(input_ids)
                predicted_value = values[:, -1]
                predicted_value_test = values[:, -2]
                loss = 0.5*(predicted_value-gt)**2
                loss_test = 0.5 * (predicted_value - og_gt) ** 2
                loss_list.append(loss)
                loss_list_test.append(loss_test)
        val_loss = torch.stack(loss_list).mean().item()
        val_loss_test = torch.stack(loss_list_test).mean().item()
        vas_trainer.accelerator.log({"val/val_loss": val_loss, "val/val_loss_test": val_loss_test}, step=None)
        vas_trainer.model.train()

# for beta in [0,1,2,3,4,5,6,7,8,9,10]:
#     for _epoch, batch in tqdm(enumerate(vas_trainer.dataloader)):
#         query_tensors = batch["query"]
#         with torch.no_grad():
#             sequences = vas_trainer.generate(
#                 query_tensors, batch_size=1, return_prompt=True, vas_generation=True, beta=beta, **generation_kwargs)
#         texts = tokenizer.batch_decode(sequences)
#         rewards = []
#         for text in texts:
#             inputs_ids = reward_tokenizer.encode(text, return_tensors='pt').to(reward_model.device)
#             reward_outputs = reward_model(inputs_ids)
#             reward = reward_outputs.logits[0]
#             rewards.append(reward)
#         mean_reward = torch.mean(torch.stack(rewards)).item()
#         print(f'beta {beta}, mean_reward {mean_reward}')
#         break

vas_trainer.model.is_peft_model = True
vas_trainer.save_pretrained("/data/scratch-oc40/pulkitag/idanshen/trl/debug")

