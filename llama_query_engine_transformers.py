from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import os 
import torch

# REF: https://github.com/nicknochnack/Falcon40B/blob/main/Falcon.ipynb

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 --upgrade
# pip install langchain einops accelerate transformers bitsandbytes

# NOTE: I was prompted to install these but it's likely that I just didn't run the above blank chain installation.
#   pip install accelerate
#   pip install optimum
#   pip install auto-gptq

# Check if cuda is available 
torch.cuda.is_available()
# Define Model ID
model_path_and_id = "TheBloke/MythoMax-L2-13B-GPTQ" #  "Open-Orca/Mistral-7B-OpenOrca"
model_file = "mythomax-l2-13b.Q4_K_M.gguf" # "mistral-7b.Q4_K_M.gguf"
# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path_and_id)
# Load Model 
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="TheBloke/MythoMax-L2-13B-GPTQ", model_file="mythomax-l2-13b.Q4_K_M.gguf",
    cache_dir='./workspace/', device_map="auto", offload_folder="offload") # torch_dtype=torch.bfloat16
# Set PT model to inference mode
model.eval()
# Build HF Transformers pipeline 
pipeline = transformers.pipeline(
    "text-generation", 
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_length=400,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

# Test out the pipeline
pipeline('who is kim kardashian?')