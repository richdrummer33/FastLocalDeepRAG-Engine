#####################################

# Post: https://stackoverflow.com/questions/76495666/running-llm-on-a-local-server
# Tutorial: https://huggingface.co/docs/transformers/installation#offline-mode

#####################################

# Hugging face transformer classes for generation
# NOTE: GPU supported (for GGML-formatted models) by the new *c*transformers lib 
# "Python bindings for the Transformer models implemented in C/C++ using GGML library" (https://github.com/marella/ctransformers)
from ctransformers import AutoModelForCausalLM, AutoTokenizer # TextStreamer (not avail in ctransformers, but is avail in transformers)
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer as AutoTokenizerHF
# Streaming ability
# from transformers import pipeline
# For datatype conversion (e.g. torch tensors with float32, float16, etc.)
import torch # REQUIRED ?
# sleep is the best meditation
import time

# llama2 or other model weights file (aka model ID)
model_path = "TheBloke/MythoMax-L2-13B-GPTQ" #  "Open-Orca/Mistral-7B-OpenOrca"
model_file = "mythomax-l2-13b.Q4_K_M.gguf" # "mistral-7b.Q4_K_M.gguf"
# Required for Llama models only (you need to generate one in your hugging face account, and request access to llama via Meta's site)
# auth_token = "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"

print("\033[95m\nDL Tokenizer...\n\033[0m")
tokenizer = AutoTokenizerHF.from_pretrained(model_path) # "use_fast" not avail in ctransformers. Probably cause it's built in c, lol
print("\033[95m\nDL Model...\n\033[0m")
model = AutoModelForSeq2SeqLM.from_pretrained(model_path) # , model_file="mythomax-l2-13b.Q4_K_M.gguf") #, hf=True ?? 
# model = AutoModelForCausalLM.from_pretrained(model_path_or_repo_id=model_path, device_map="auto", trust_remote_code=False, revision="main")

print("\033[95m\nSaving files...\n\033[0m")
tokenizer.save_pretrained(model_path)
model.save_pretrained(model_path)

print("\033[95m\nLoading Tokenizer...\n\033[0m")
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("\033[95m\nLoading Model...\n\033[0m")
model = AutoModel.from_pretrained(model_path)


# wait a sec so you can process what just happened
time.sleep(1)

prompt = "Tell me about AI"
prompt_template=f'''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:

'''

print("\033[95m\nSending prompt...\n\033[0m")
tokens = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()

output = model._llm.generate(tokens=tokens, temperature=0.7, top_p=0.95, top_k=40)
print(tokenizer.decode(output[0]))

# Inference can also be done using transformers' pipeline
# print("*** Pipeline:")
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=512,
#     do_sample=True,
#     temperature=0.7,
#     top_p=0.95,
#     top_k=40,
#     repetition_penalty=1.1
# )
# 
# print(pipe(prompt_template)[0]['generated_text'])
