# WORKS!!!
import winsound
import torch
from llama_index import (
    KeywordTableIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext
)

#####################################################################
# Misc classes
#####################################################################

class NotificationType:
    WARNING = "C:\\Windows\\Media\\Windows Exclamation.wav"
    SUCCESS = "C:\\Windows\\Media\\Speech On.wav"

def play_notification_sound(notification_type):
    if notification_type == NotificationType.WARNING:
        sound_path = NotificationType.WARNING
    elif notification_type == NotificationType.SUCCESS:
        sound_path = NotificationType.SUCCESS
    winsound.PlaySound(sound_path, winsound.SND_FILENAME)

# name attr accomodation
from ctransformers import AutoModelForCausalLM
class MyAutoModelForCausalLM(AutoModelForCausalLM):
    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metadata = {'name': name}  # Initialize metadata with name

    @property
    def metadata(self):
        # Return the metadata dictionary
        return self._metadata

    def __getattr__(self, name: str):
        # Since metadata is now a regular attribute, we don't need to override __getattr__ for it
        return super().__getattr__(name)


#####################################################################
#####################################################################


#### Fields and Definitions
#model_llm_path = "./models/Mistral-7B-Instruct-v0.1"
#model_name = "mistralai/Mistral-7B-Instruct-v0.1"
model_path = "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
model_embeddings_path = "./sentence-transformers/all-mpnet-base-v2"
data_path = "C:/Users/richd/Desktop/test-rag" # "D:/Git/ebook-GPT-translator-refined"
config_llm_path = "./models/Mistral-7B-Instruct-v0.1/config.json"
# set context window size and number of output tokens
context_window = 2048
num_output = 256

#### LLM load
from ctransformers import AutoConfig, Config
from ctransformers.llm import LLM
print("\033[95m\nLoading Model...\n\033[0m")
conf = AutoConfig(Config(temperature=0.8, repetition_penalty=1.1,
                         batch_size=52, max_new_tokens=2048,
                         context_length=2048, gpu_layers=50,
                         stream=True))
llm = AutoModelForCausalLM.from_pretrained(model_path_or_repo_id=model_path, model_type="mistral", gpu_layers=50) #  model_type="mistral" 
#llm = AutoModelForCausalLM.from_pretrained(model_llm_path, from_tf=True) # config = conf, model_type="mistral" 


#### Embeddings and service context
########################################################
## FOR LLAMA CPP (GGUF COMPAT): https://gpt-index.readthedocs.io/en/latest/examples/llm/llama_2_llama_cpp.html
########################################################
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding   # embedding
print("\033[95m\nEmbeddings...\n\033[0m")
embed_model = HuggingFaceEmbeddings(model_name=model_embeddings_path)
embeddings = LangchainEmbedding(embed_model, model_name=model_embeddings_path)


#### Set things up for indexer
print("\033[95m\nIndexing...\n\033[0m")
from llama_index import ServiceContext
from llama_index import set_global_service_context
service_context = ServiceContext.from_defaults(
    chunk_size=512,
    llm=llm,
    embed_model=embeddings
)
set_global_service_context(service_context) 


### Do docs indexing!
from llama_index import VectorStoreIndex, SimpleDirectoryReader
documents = SimpleDirectoryReader(data_path).load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
index.storage_context.persist()


## set up query engine...
print("\033[95m\nQuery engine...\n\033[0m")
query_engine = index.as_query_engine()
play_notification_sound(NotificationType.SUCCESS)


### Ready!
while True:
    prompt = input("\nEnter prompt: ")
    mistral_prompt = f"<s>[INST] {prompt} [/INST]"

    print("\033[95m\nGenerating output from promt...\n\033[0m")

    for answer in llm(mistral_prompt, temperature=0.8, repetition_penalty=1.1, 
                         batch_size=52, max_new_tokens=2048, stream=True):
        print(answer, end="", flush=True)
