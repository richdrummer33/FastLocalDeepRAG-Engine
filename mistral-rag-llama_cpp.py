# WORKS!!!
import winsound
import torch

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
data_path = "D:\Git\EscapeRoom3DGitLab\Assets\Scripts" # "C:/Users/richd/Desktop/test-rag" # "D:/Git/ebook-GPT-translator-refined"
config_llm_path = "./models/Mistral-7B-Instruct-v0.1/config.json"
# set context window size and number of output tokens
context_window = 2048
num_output = 256

#### LLM load
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
print("\033[95m\nLoading Model...\n\033[0m")
llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url=None,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=model_path,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 50},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)


#### Embeddings and service context
########################################################
## FOR LLAMA CPP (GGUF COMPAT): https://gpt-index.readthedocs.io/en/latest/examples/llm/llama_2_llama_cpp.html
########################################################
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
print("\033[95m\nEmbeddings...\n\033[0m")
embed_model = HuggingFaceEmbeddings(model_name=model_embeddings_path)


#### Set things up for indexer
print("\033[95m\nIndexing...\n\033[0m")
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    set_global_service_context
)
service_context = ServiceContext.from_defaults(
    chunk_size=2048,
    llm=llm,
    embed_model=embed_model
)
set_global_service_context(service_context) 


### Do docs indexing!
from llama_index import VectorStoreIndex, SimpleDirectoryReader
documents = SimpleDirectoryReader(data_path).load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
index.storage_context.persist()


## set up query engine...
print("\033[95m\nQuery engine...\n\033[0m")
query_engine = index.as_query_engine(streaming=True)
play_notification_sound(NotificationType.SUCCESS)


### Ready!
while True:
    prompt = input("\nEnter prompt: ")
    mistral_prompt = f"<s>[INST] {prompt} [/INST]"
    response = query_engine.query(prompt)
    print("\033[95m\nGenerating output from promt...\n\033[0m")
    
    response.print_response_stream()
    #print(str(response))