# command: deepspeed --num_gpus 1 blip2_deepspeed_manual_config.py

from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration, Blip2Config
import torch
import os
import gc
import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig
import torch.distributed as dist




# processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto")

# img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# question = "how many dogs are in the picture?"
# inputs = processor(raw_image, question, return_tensors="pt").to("cuda")

# out = model.generate(**inputs)
# print(processor.decode(out[0], skip_special_tokens=True).strip())


from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration, Blip2Config
import torch
import deepspeed

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
# )

ds_config = {
    "zero": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 50_000_000,
        "stage3_prefetch_bucket_size": 0.9 * 50_000_000,
        "stage3_param_persistence_threshold": 0,
        "offload_param": {
            "device": "cpu",
            "pin_memory": False
        }
    },
}


ds_config = {
    "fp16": {
        "enabled": True,
    },
    "bf16": {
        "enabled": False,
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 50_000_000,
        "stage3_prefetch_bucket_size": 0.9 * 50_000_000,
        "stage3_param_persistence_threshold": 0,
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        }
    },
    "train_batch_size": 4,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False,
}

dschf = HfDeepSpeedConfig(ds_config)

torch.cuda.empty_cache()
gc.collect()
deepspeed.runtime.utils.see_memory_usage("pre-from-pretrained", force=True)

model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
)

deepspeed.runtime.utils.see_memory_usage("post-from-pretrained", force=True)

model = model.eval()

ds_engine = deepspeed.initialize(model=model, config_params=ds_config)
ds_engine = ds_engine[0]

model = ds_engine.module

# model.to(device)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "Question: how many cats are there? Answer:"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)


generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)


local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

train_batch_size = 1 * world_size

deepspeed.init_distributed("nccl")
rank = dist.get_rank()

def print_rank0(*msg):
    if rank != 0:
        return
    print(*msg)

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")


ds_config = {
    "fp16": {
        "enabled": True,
    },
    "bf16": {
        "enabled": False,
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 50_000_000,
        "stage3_prefetch_bucket_size": 0.9 * 50_000_000,
        "stage3_param_persistence_threshold": 0,
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        }
    },
    "train_batch_size": train_batch_size,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False,
}

dschf = HfDeepSpeedConfig(ds_config)

torch.cuda.empty_cache()
gc.collect()
deepspeed.runtime.utils.see_memory_usage("pre-from-pretrained", force=True)

model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
)

deepspeed.runtime.utils.see_memory_usage("post-from-pretrained", force=True)

model = model.eval()

# ds_engine = deepspeed.init_inference(model, mp_size=1, dtype=torch.float16, replace_with_kernel_inject=True, config=ds_config)
ds_engine = deepspeed.initialize(model=model, config_params=ds_config)
ds_engine = ds_engine[0]

ds_engine.module.eval()
model = ds_engine.module

# model.to(device)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert('RGB')

prompt = "Question: how many cats are there? Answer:"
inputs = processor(images=[image, image, image, image], text=[prompt, prompt, prompt, prompt], return_tensors="pt").to(device, torch.float16)


generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)