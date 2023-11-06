# WIP
# IDEATION: "AI Tools and Project Ideas" in obsidian projet
# GPT CHAT: https://chat.openai.com/share/2633e029-fb49-4734-b132-c5fb013708ba

import os
from sentence_transformers import SentenceTransformer
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ListIndex
from ctransformers import AutoModelForCausalLM, AutoConfig, Config
import winsound
import torch

# >>> FIELDS >>>
root_directory = "C:/Users/richd/Desktop/test-rag" # "D:/Git/ebook-GPT-translator-refined"
persist_dir = "./indexed-data"
required_exts = [".md", ".py", ".txt", ".json"]

# >>> CLASSES >>>
class NotificationType:
    WARNING = "C:\\Windows\\Media\\Windows Exclamation.wav"
    SUCCESS = "C:\\Windows\\Media\\Speech On.wav"

def play_notification_sound(notification_type):
    if notification_type == NotificationType.WARNING:
        sound_path = NotificationType.WARNING
    elif notification_type == NotificationType.SUCCESS:
        sound_path = NotificationType.SUCCESS
    winsound.PlaySound(sound_path, winsound.SND_FILENAME)

def read_files_recursively(root_dir):
    sentences = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                sentences.append(file.read())
    return sentences


import os
from llama_index import StorageContext
from llama_index import load_index_from_storage
#from llama_index.docstore import SimpleDocumentStore, SimpleVectorStore, SimpleIndexStore
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader


def get_saved_indexed():

    try:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    except Exception as e:
        print("\033[95mNo index file saved\033[0m")
        return None

    # have index. Don't need to specify index_id if there's only one index in storage context
    print("\033[95mIndex file exists. Loading it...\033[0m")
    index = load_index_from_storage(storage_context) 

    return index


def index_data(service_context):
    
    # Get docs and embed
    print("\033[95mGenerating embeddings...\033[0m")

    embeddings = SimpleDirectoryReader(
        root_directory, 
        recursive=True,
        required_exts = required_exts
        ).load_data()

    # Indexing with Llama Index
    print("\033[95mGenerating vector store with service context llm...\033[0m")

    index = VectorStoreIndex.from_documents(embeddings, service_context=service_context) # VectorStoreIndex.from_documents(embeddings)
    
    return index

#######################################################################################
# LLM for querying
# https://gpt-index.readthedocs.io/en/v0.6.27/how_to/customization/custom_llms.html
#######################################################################################

# Using a Custom LLM Model - Advanced
from llama_index import LLMPredictor, ServiceContext
from transformers import pipeline
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any


# set context window size and number of output tokens
context_window = 2048
num_output = 256

print("\033[95mStarting pipeline...\033[0m")
model_name = "./models/Mistral-7B-Instruct-v0.1"
pipeline = pipeline("text-generation", model=model_name, device=0, model_kwargs={"torch_dtype":torch.bfloat16})

class CustomLLM(LLM):
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt_length = len(prompt)
        response = pipeline(prompt, max_new_tokens=num_output)[0]["generated_text"]

        # only return newly generated tokens
        return response[prompt_length:]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"

# define our LLM
print("\033[95mInit LLM Predictor...\033[0m")
llm_predictor = LLMPredictor(llm=CustomLLM())

print("\033[95mInit service context...\033[0m")
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, 
    context_window=context_window, 
    num_output=num_output
)

#######################################################################################
#######################################################################################

# try load prev saved indexed data
print("\033[95mTry and get saved indexed data...\033[0m")
index = get_saved_indexed()

# index the dir!
if index is None:
    print("\033[95mIndexing new data...\033[0m")
    index = index_data(service_context)
    index.storage_context.persist(persist_dir = persist_dir)

# ....Query!
query_engine = index.as_query_engine()
play_notification_sound(NotificationType.SUCCESS)
while True:
    prompt = input("\nEnter prompt: ")
    mistral_prompt = f"<s>[INST] {prompt} [/INST]"
    
    print("\033[94m\nQuerying indexed documents...\n\033[0m")
    response = query_engine.query(mistral_prompt)

    print("\033[94m\n" + str(response) + "\n\033[0m")
