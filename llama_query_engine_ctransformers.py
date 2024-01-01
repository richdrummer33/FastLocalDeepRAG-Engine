#####################################

# >>> README >>>
# NOTE 1: 
#   This project started here... https://www.youtube.com/watch?v=hMJgdVJWQRU&list=PLmy6kHlcQI1S2Wh0jlyclwrHsVA_-QmuJ&index=47&t=290s
#   This vid goes over everything, but slightly differs as it uses another model format
#
# NOTE 2: 
#   The facebook team released the GGUF model format as a replacement to GGML on 21 August 2023
#   See details here: https://medium.com/@phillipgimmi/what-is-gguf-and-ggml-e364834d241c

### INSTALLATIONS ###
# pip install ctransformers[gptq] # apparently required for GGUF models
# pip install transformers
# wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf
# Also can download Hugging Face models manually from LM Studio

### REFERENCES ###
# USE THIS TUTORIAL:
#   https://artificialcorner.com/run-mistral7b-quantized-for-free-on-any-computer-2cadc18b45a2
#
# HF Tutorial for Mistral 7b:
#   https://huggingface.co/docs/transformers/main/en/model_doc/mistral
#
# "A complete guide to running local LLM models"
#   https://bootcamp.uxdesign.cc/a-complete-guide-to-running-local-llm-models-3225e4913620
# 
# Tutorial with ctransformers and Llama 2 ggml model:
#   https://vilsonrodrigues.medium.com/run-llama-2-models-in-a-colab-instance-using-ggml-and-ctransformers-41c1d6f0e6ad
# 
# GGML model transformer library:
#   "A Python library with GPU accel, LangChain support, and OpenAI-compatible AI server"
#   https://github.com/marella/ctransformers    
# 
# GGUF from python:
#   https://github.com/abetlen/llama-cpp-python
# 
# [FOR REF ONLY] GGUF source project (llama.cpp)
#   "The source project for GGUF. Offers a CLI and a server option"
#   https://github.com/ggerganov/llama.cpp
# 
# Other libs known to support GGUF models listed here: https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF

### MODELS ###
# Local Model Comparisons: 
#   https://github.com/Troyanovsky/Local-LLM-Comparison-Colab-UI
#
# Mistral 7b:
#   "This model is proficient at both roleplaying and storywriting due to its unique nature"
#   https://www.reddit.com/r/LocalLLaMA/comments/16twtfn/llm_chatrp_comparisontest_mistral_7b_base_instruct/
# 
# MythoMax
#   "This model is proficient at both roleplaying and storywriting due to its unique nature"
#   ctransformers? Apparently so (see link below)
#   https://huggingface.co/TheBloke/MythoMax-L2-13B-GPTQ

### PERFORMANCE ###
# Hardware Acceleration: Basic Linear Algebra Subprograms (BLAS)
#   [How to activate BLAS?] https://github.com/ggerganov/llama.cpp/issues/627
#   https://github.com/abetlen/llama-cpp-python
#   https://github.com/OpenMathLib/OpenBLAS

### INTRUCTION TEMPLATES ###
# Mistral 7b user:
# (from reddit post here: https://www.reddit.com/r/LocalLLaMA/comments/16y6r3x/a_7b_better_than_llama_65b_now_mistral_orca_is_out/)
#   user: <|im_start|>user
#   bot: |-
#     <|im_end|>
#     <|im_start|>assistant
#   turn_template: '<|user|>\n<|user-message|>\n<|bot|>\n<|bot-message|><|im_end|>\n'
# 
# Mistral 7b system/user 
# (from gguf model reference here: https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF)
#   <|im_start|>system
#   {system_message}<|im_end|>
#   <|im_start|>user
#   {prompt}<|im_end|>
#   <|im_start|>assistant

#####################################

# for playing notification sounds
import winsound

class NotificationType:
    WARNING = "C:\\Windows\\Media\\Windows Exclamation.wav"
    SUCCESS = "C:\\Windows\\Media\\Speech On.wav"

def play_notification_sound(notification_type):
    if notification_type == NotificationType.WARNING:
        sound_path = NotificationType.WARNING
    elif notification_type == NotificationType.SUCCESS:
        sound_path = NotificationType.SUCCESS
    winsound.PlaySound(sound_path, winsound.SND_FILENAME)

# Hugging face transformer classes for generation
# NOTE: GPU supported (for GGML-formatted models) by the new *c*transformers lib 
# "Python bindings for the Transformer models implemented in C/C++ using GGML library" (https://github.com/marella/ctransformers)
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig # TextStreamer (not avail in ctransformers, but is avail in transformers)\
from transformers import pipeline
import time # sleep is the best medicine
device = "cuda" # the device to load the model onto

# ********* apparently mistral-7b-instruct-v0.1.Q5_K_M.gguf works *********
# model_name_or_path = "TheBloke/MythoMax-L2-13B-GGUF" # MythoMax-L2-13B-GGUF" #  "TheBloke/Mistral-7B-OpenOrca"
# model_file = "mythomax-l2-13b.Q4_K_M.gguf" # "mistral-7b.Q4_K_M.gguf"

# >>> SETUP >>>
print("\033[95m\nLoading Model...\n\033[0m")
# REF: https://huggingface.co/docs/transformers/main/en/model_doc/mistral
try:
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="./mistralai/Mistral-7B-v0.1")
    tokenizer = AutoTokenizer.from_pretrained("./mistralai/Mistral-7B-v0.1", use_fast=True)
    print("\033[92m\nModel Loaded!\n\033[0m")
    play_notification_sound(NotificationType.SUCCESS)
except:
    print("\033[91m\nModel failed to load!\n\033[0m")
    play_notification_sound(NotificationType.WARNING)
    time.sleep(5)
    exit()

# >>> PROMPT >>>
prompt = "Tell me about AI"
prompt_template=f'''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:

'''

# >>> QUERY >>>
print("\033[95m\nLoading tokenizer!\n\033[0m")
model_inputs = tokenizer([prompt], return_tensors='pt').to(device) # .input_ids.cuda()
print("\033[92m\nTokenizer Loaded! Model to GPU\n\033[0m")
model.to(device) # REF: https://huggingface.co/docs/transformers/main/en/model_doc/mistral

print("\033[95m\nModel generating output from promt...\n\033[0m")
generated_ids = model.generate(**model_inputs, max_new_tokens=150, temperature=0.7, do_sample=True)
print("\033[92m\nOutput generated! Decoding output tokens...\n\033[0m")
tokenizer.batch_decode(generated_ids)[0] # REF: https://huggingface.co/docs/transformers/main/en/model_doc/mistral
print("\033[92m\nOutput decoded!\n\033[0m")

play_notification_sound(NotificationType.WARNING)
print("\033[95m\nOutput:\n\033[0m")
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
