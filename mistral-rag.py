# WIP
# ref: https://github.com/mickymult/RAG-Mistral7b/blob/main/RAG_testing_mistral7b.ipynb
# see "AI Tools and Project Ideas" in obsidian projet for ideations

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM

documents = SimpleDirectoryReader("C:/Users/richd/Desktop/test-rag", recursive=True).load_data()

from llama_index.prompts.prompts import SimpleInputPrompt
system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."
# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = "<|USER|>{query_str}<|ASSISTANT|>"

####### ?????? #######
# To login, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
# !huggingface-cli login
####### ?????? #######

import torch
torch.set_default_device('cuda')

from ctransformers import AutoModelForCausalLM, AutoConfig, Config

# access_token = "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"

# print("\033[95m\nLoading Model...\n\033[0m")
# conf = AutoConfig(Config(temperature=0.8, repetition_penalty=1.1, 
#                         batch_size=52, max_new_tokens=2048, 
#                         context_length=2048, gpu_layers=50,
#                         stream=True))

# llm = AutoModelForCausalLM.from_pretrained("./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
#                                            model_type="mistral", config = conf, token=access_token)

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.1, "do_sample": True},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="./models/Mistral-7B-Instruct-v0.1",
    model_name="./models/Mistral-7B-Instruct-v0.1",
    device_map="auto",
    tokenizer_kwargs={"max_length": 4096},
    model_kwargs={"torch_dtype": torch.float16}
)

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext

print("\033[95m\nGenerating embeddings...\n\033[0m")
embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="./sentence-transformers/all-mpnet-base-v2")
)

print("\033[95m\nInit service context...\n\033[0m")
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)

print("\033[95m\nGenerating vector store...\n\033[0m")
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# confirm docs ingested
# for doc in index.docstore.list.len:
#     print(doc.name)

print("\033[95m\nBegin query engine!\n\033[0m")
query_engine = index.as_query_engine()
# response = query_engine.query("what was the revenuew from aws in 2022?")
# print(response)

print("\033[95m\nStarting prompts loop...\n\033[0m")
while True:
    prompt = input("\nEnter prompt: ")
    # prompt = f"<s>[INST] {prompt} [/INST]"

    print("\033[95m\nGenerating output from promt...\n\033[0m")
    response = query_engine.query(prompt)
    print(response)

    # stream?
    # for answer in llm(mistral_prompt, temperature=0.8, repetition_penalty=1.1, 
    #                      batch_size=52, max_new_tokens=2048, stream=True):
    #     print(answer, end="", flush=True)