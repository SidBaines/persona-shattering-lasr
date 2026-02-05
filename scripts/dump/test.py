# %%
from huggingface_hub import login

access_token = input("Paste your Hugging Face token here: ")
login(token=access_token)

# %%

### Delete HF cache
# from huggingface_hub import scan_cache_dir

# cache_info = scan_cache_dir()

# revisions_to_delete = []
# for repo in cache_info.repos:
#     revisions_to_delete.extend([rev.commit_hash for rev in repo.revisions])

# # Delete all revisions of that repo
# delete_strategy = cache_info.delete_revisions(*revisions_to_delete)
# delete_strategy.execute()
# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

REPO      = "maius/llama-3.1-8b-it-personas"  
PERSONA   = "sarcasm"                         
BASE_ID   = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
base = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(base, REPO, subfolder=PERSONA)

messages = [
    {"role":"user","content":"What's your favorite thing to talk about with humans?"}
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=256, temperature=0.7, top_p=0.9, top_k=None, min_p=0.0)
print(tokenizer.decode(out[0], skip_special_tokens=True))


# %%
model.device
# %%
base.device
# %%
inputs
# %%
dir(model)
# %%
model.base_model
# %%
model.base_model.generate(inputs, max_new_tokens=256, temperature=0.7, top_p=0.9, top_k=None, min_p=0.0)

# %%
dir(model.base_model)
# %%
dir(model.base_model.model)
# %%
model.base_model.model.generate(inputs, max_new_tokens=256, temperature=0.7, top_p=0.9, top_k=None, min_p=0.0)
# %%
dir(model.base_model.model)

# %%
model(inputs["input_ids"])
# %%
inputs
# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

REPO      = "maius/llama-3.1-8b-it-personas"  
PERSONA   = "sarcasm"                         
BASE_ID   = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
base = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    device_map="cuda",
    torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(base, REPO, subfolder=PERSONA)

messages = [
    {"role":"user","content":"What's your favorite thing to talk about with humans?"}
]
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
# inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
attention_mask = torch.ones_like(input_ids)
out = model.generate(input_ids, max_new_tokens=256, temperature=0.7, top_p=0.9, top_k=None, min_p=0.0, attention_mask=attention_mask)
print(tokenizer.decode(out[0], skip_special_tokens=True))
# %%
