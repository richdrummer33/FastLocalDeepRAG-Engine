# WORKS!!!
import winsound
# import torch
import time

import torch.cuda as cuda # for tracking gpu memory usage

# CUDA / cuBLAS check
# print(torch.cuda.is_available())  # Check if CUDA is available
# print(torch.backends.cudnn.enabled)  # Check if cuDNN is enabled, which indirectly implies cuBLAS is working

#####################
### Fields and Definitions
#####################

model_path = "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
model_embeddings_path = "./sentence-transformers/all-mpnet-base-v2"
data_path = "./data/" # "D:/Git/Unseen/Assets/Code/" # "D:\\Git\\twinny-api" #  #"D:/Git/EscapeRoom3DGitLab/Assets/Scripts" #"C:/Users/richd/Desktop/test-rag" #"D:/Git/ebook-GPT-translator-refined"
#config_llm_path = "./models/Mistral-7B-Instruct-v0.1/config.json" 

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# MISC CLASSES
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# print version llama_index
from llama_index import __version__
print("llama_index version: " + __version__)

class NotificationType:
    WARNING = "C:\\Windows\\Media\\Windows Exclamation.wav"
    SUCCESS = "C:\\Windows\\Media\\Speech On.wav"

def play_notification_sound(notification_type):
    if notification_type == NotificationType.WARNING:
        sound_path = NotificationType.WARNING
    elif notification_type == NotificationType.SUCCESS:
        sound_path = NotificationType.SUCCESS
    winsound.PlaySound(sound_path, winsound.SND_FILENAME)

def user_select_data_path():
    global data_path
    choice = input("\033[95m\nPress 1 to use the default data-path, or 2 to pick one: \n\033[0m")

    if(choice != "1"):
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        data_path = filedialog.askdirectory()

    print("\033[95m\nData path: " + data_path + "\n\033[0m")

# Function to print the current and peak memory usage (doesnt seem to work)
# def print_gpu_memory_usage():
#     print("Current GPU Memory Allocated:", torch.cuda.memory_allocated())
#     print("Peak GPU Memory Allocated:", torch.cuda.max_memory_allocated())

import re
from typing import Tuple, List

import re
from typing import Tuple, List
from itertools import zip_longest

import re
from typing import Tuple, List
from itertools import zip_longest

def parse_choice_select_answer_fn(
    answer: str, num_choices: int, raise_error: bool = False
) -> Tuple[List[int], List[float]]:
    """Parse choice select answer function."""
    answer_lines = answer.split("\n")
    answer_nums = []
    answer_relevances = []

    print("parsing lines: \n" + str(answer_lines))

    # Temporary storage for document numbers
    temp_doc_nums = []
    # Temporary storage for relevance scores
    temp_relevances = []

    for answer_line in answer_lines:
        # Check for document lines and extract the number
        doc_match = re.match(r'Document (\d+):', answer_line)
        if doc_match:
            temp_doc_nums.append(int(doc_match.group(1)))

        # Check for relevance score line and split the scores
        relevance_match = re.match(r'Relevance score: (.+)', answer_line)
        if relevance_match:
            scores_str = relevance_match.group(1)
            temp_relevances = [float(score.strip()) for score in scores_str.split(',')]

    # Match documents with relevance scores
    for doc_num, relevance in zip_longest(temp_doc_nums, temp_relevances, fillvalue=0.0):
        answer_nums.append(int(doc_num))
        answer_relevances.append(int(relevance))

    log_opt = ""
    if not answer_nums:
        if raise_error:
            raise ValueError("No valid answer numbers found.")
        else:
            print("No valid answer numbers found.")
            answer_nums.append(0)
            answer_relevances.append(int(0))
    if len(answer_nums) == 0:
        print("No answer nums added from relevance matching. Adding top 3.")
        log_opt = "(top 3)"
        answer_nums = answer_nums[:3]
        answer_relevances = answer_relevances[:3]
    
    relevance_override = 10
    for x in answer_relevances:
        if x == 0:
            answer_relevances[answer_relevances.index(x)] = int(relevance_override)
        relevance_override -= int(1)

    
    print("\nanswer_nums " + log_opt + ": " + str(answer_nums))
    print("answer_relevances " + log_opt + ": " + str(answer_relevances) + "\n")

    return answer_nums, answer_relevances


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

    def list_all_doc_key_values(index_summary: DocumentSummaryIndex) -> List[Dict[str, str]]:
        """
        List all document key-value pairs from the provided IndexDocumentSummary instance.
        Returns a list of dictionaries, each containing 'doc_id' and 'summary_id'.
        """
        all_key_values = []
        for doc_id, summary_id in index_summary.doc_id_to_summary_id.items():
            all_key_values.append({'doc_id': doc_id, 'summary_id': summary_id})
            print(f"doc_id: {doc_id}, summary_id: {summary_id}")

        return all_key_values

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# PROGRAM
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# confirm that data_path exists
import os
if not os.path.exists(data_path):
    print("data_path does not exist")
    exit()

use_gpt = False
use_doc_summary = False

SUMMARY_QUERY = (
    """You are a Unity developer. Write a class summary:
    Class Definition: Name, base class, and interfaces (if applicable).
    Class Role: Define the class's functionality. If applicable, also note any significant class references, dependents or dependencies.
    Methods: List all method names, with attributes in square brackets if applicable.
    Features: Take note of any special features such as Photon RPC calls"""
)

#####################
### Load/init the local gguf LLM (via llama cpp)
#####################
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
if not use_gpt:
    print("\033[95m\nLoading LLM model...\n\033[0m")
    print_gpu_memory_usage()
    llm = LlamaCPP(
        # You can pass in the URL to a GGML model to download it automatically
        model_url=None,
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path=model_path,
        temperature=0.1,
        max_new_tokens=256,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=5500,
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
    print("\033[95m\nLLM model Loaded!\n\033[0m")
    print_gpu_memory_usage()
else:
    import openai
    import os
    openai.api_key = os.environ["OPENAI_API_KEY"]
    from llama_index.llms import OpenAI
    llm = OpenAI(temperature=0.1, model="gpt-4")

# press enter to continue in green colored text
print("\033[92m\nPress enter to continue...\n\033[0m")

###################################
### Embeddings and service context
### NOTE FOR LLAMA CPP (GGUF COMPAT): https://gpt-index.readthedocs.io/en/latest/examples/llm/llama_2_llama_cpp.html
###################################
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
embed_model = None

def load_embeddings_model():
    global embed_model
    print("\033[95m\nLoading embeddings model...\n\033[0m")
    print_gpu_memory_usage()

    model_kwargs={"device": "cuda:0"}
    embed_model = HuggingFaceEmbeddings(model_name=model_embeddings_path, model_kwargs=model_kwargs) #, model_kwargs=model_kwargs) #device = cuda
    print("\033[95m\nEmbeddings model loaded!\n\033[0m")
    print_gpu_memory_usage()
    
    # press enter to continue in green colored text
    print("\033[92m\nPress enter to continue...\n\033[0m")

##############################
### Set things up for indexer
### NOTE: https://blog.llamaindex.ai/a-new-document-summary-index-for-llm-powered-qa-systems-9a32ece2f9ec
##############################
from llama_index import (
    SimpleDirectoryReader,
    LLMPredictor, # added for doc summary 
    VectorStoreIndex,
    ServiceContext,
    set_global_service_context
)
# Set up response synth - generates response from llm for a user query and a given set of text chunks
# NOTE: https://gpt-index.readthedocs.io/en/latest/module_guides/querying/response_synthesizers/root.html
from llama_index.response_synthesizers import ResponseMode, get_response_synthesizer
# For synthesizing summaries for the docs https://blog.llamaindex.ai/...
response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.TREE_SUMMARIZE) # EG. USAGE: response = response_synthesizer.synthesize("query text", nodes=[Node(text="text"), ...])


###########################
### Set up service context
# Defines the llm and embed models and chunk size to retreive
# NOTE: Memory-pool issues on docs ingestion: https://github.com/imartinez/privateGPT/issues/181
#       Should I also increase LLM context window to avoid? Tradeoffs with performance/batching?
###########################
service_context = None

def init_service_context():
    global service_context
    global embed_model

    if embed_model is None:
        load_embeddings_model()

    service_context = ServiceContext.from_defaults(
        chunk_size=400, # 1024 # 2048 (I think this too large & caused mem errors on ingest)
        chunk_overlap=40, # 10%
        llm=llm,
        embed_model=embed_model
    )
    # set_global_service_context(service_context) # Necessary? Maybe not 

###################
### Index the docs
###################
from llama_index import VectorStoreIndex, SimpleDirectoryReader, DocumentSummaryIndex, StorageContext
from llama_index.indices.loading import load_index_from_storage

build_new_index = input("\033[95m\nPress 1 to load stored index. Press 2 to reimport index from storage: \n\033[0m")

# Option 1: Load index from storage
if(build_new_index.lower() == "1"):
    try:
        # we are going to use the embeddings model to load the index from storage
        load_embeddings_model()
        # load the index from storage
        storage_context = StorageContext.from_defaults(persist_dir="./index")
        # initialize/instantiate the indexer with the indexed data just fetched from storage
        doc_summary_index = load_index_from_storage(
            storage_context,
            embed_model=embed_model,
            llm_predictor=LLMPredictor(llm)
        )
        print("\033[95m\nLoaded index from storage...\n\033[0m")
    except:
        print("\033[95m\nFailed to load index from storage...\n\033[0m")
        build_new_index = None

    # Init service context for the retreiver (llm OR embedding-based cosine similarity) query engine
    init_service_context()

# Option 2: Build new index
else:
    # What folder root to index?
    user_select_data_path()
    # Init service context for the indexer AND retreiver (llm OR embedding-based cosine similarity) query engine
    init_service_context()
    print("\033[95m\nFetch documents...\n\033[0m")
    # Fetch documents from the data_path
    reader  = SimpleDirectoryReader(data_path, recursive=True, exclude=['*.meta', '*.preset', '*.bnk', '*.wem', '*.fbx', '*.obj', '*.wav', '*.onnx', '*.otf', '*.mat', '*.png', '*.prefab', '*.unity', '.pyc', '.pyd', '.pyo', '.dll', '.exe', '.o', '.obj', '.a', '.lib', '.so', '*.dylib']) # '*.txt', '.json']
    documents = reader.load_data()
    print("\033[95m\nIndexing documents data...\n\033[0m")
    # Index the documents
    doc_summary_index = DocumentSummaryIndex.from_documents(
        documents,
        service_context=service_context,
        summary_query=SUMMARY_QUERY,
        response_synthesizer=response_synthesizer, # will use response synth to generate llm response to retreived chunks
        show_progress=True,
        embed_model=embed_model,
    )
    # Persist the index to storage
    doc_summary_index.storage_context.persist("./index")  

##########################
### Set up docs retreiver
### REF: https://gpt-index.readthedocs.io/en/latest/examples/index_structs/doc_summary/DocSummary.html
##########################
from llama_index.indices.document_summary import DocumentSummaryIndexLLMRetriever, DocumentSummaryIndexEmbeddingRetriever
retriever = None
selection = input("Enter 1 for LLM retriever, 2 for Embedding retriever: ")

if selection == "1":
    retriever = DocumentSummaryIndexLLMRetriever(
        doc_summary_index,
        # choice_select_prompt=choice_select_prompt,
        # choice_batch_size=choice_batch_size,
        # format_node_batch_fn=format_node_batch_fn,
        # choice_batch_size=10,
        # choice_top_k=5, # 5 gave great answer!
        parse_choice_select_answer_fn=parse_choice_select_answer_fn,
        service_context=service_context
    )
else:
    retriever = DocumentSummaryIndexEmbeddingRetriever(
        doc_summary_index,
        # choice_select_prompt=choice_select_prompt,
        # choice_batch_size=choice_batch_size,
        # format_node_batch_fn=format_node_batch_fn,
        # choice_batch_size=10,
        # choice_top_k=5, # 5 gave great answer!
        parse_choice_select_answer_fn=parse_choice_select_answer_fn,
        service_context=service_context
    )

###################################
### Conversational memory
### NOTE NOTE NOTE NOTE : NOT IMPLEMENTED, UNTESTED, UNVERIFIED, UNKNOWN
###################################
#from langchain.agents import Tool
from llama_index.memory import ChatMemoryBuffer
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

conversational_memory = ConversationBufferWindowMemory(memory_key='chat_history', k=5, return_messages=True)# Initialize agent with conversational memory
#agent_executor = initialize_agent(tools, llm, agent="conversational-react-description", memory=conversational_memory)

###########################
### Set up query engine...
### Response/summarization mode can include auto-iterative prompt refinement
### refine, compact, tree_summarize, etc 
### REF (docs retreival): https://gpt-index.readthedocs.io/en/latest/examples/index_structs/doc_summary/DocSummary.html
###########################
print("\033[95m\nQuery engine...\n\033[0m")
from llama_index.query_engine import RetrieverQueryEngine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer
    #, memory=conversational_memory,
)
#query_engine = doc_summary_index.as_query_engine(response_mode=ResponseMode.TREE_SUMMARIZE, use_async=True)
play_notification_sound(NotificationType.SUCCESS)


# >>>>>>>>>>>>>>>>>>>>>>>>>
# Prompt time!
# >>>>>>>>>>>>>>>>>>>>>>>>>
from typing import Dict, List

prev_prompts = []
current_index = -1

while True:
    prompt = input("\nEnter prompt: ")
    if "list_ids" in str(prompt):
        print("\033[95m\nListing IDs...\n\033[0m")
        list_all_doc_key_values(doc_summary_index)
    elif "doc_id" in str(prompt):
        print("\033[95m\nQuerying docstore...\n\033[0m")
        doc_id = str(prompt).split("doc_id:")[1].split("\n")[0]
        doc_id = doc_id.strip()
        doc_summary = doc_summary_index.get_document_summary(doc_id)
        print(doc_summary)
    else:
        print("\033[95m\nGenerating output from promt...\n\033[0m")
        response = query_engine.query(prompt)

    canStream = False
    try:
        response.print_response_stream()
        canStream = True
    except:
        print("cannot stream")

    if not canStream:
        print("\nResponse: ")
        print(str(response))
        



##########################################
## OLD
##########################################

# Define tools
#tools = [
#    Tool(
#        name = "LlamaIndex",
#        func=lambda q: str(index.as_query_engine().query(q)),
#        description="You are a person who saerches codebases and answers questions related to that codebase in a conversational manner.",
#        return_direct=True
#    ),
#]

### Optional user verifiation of summaries
# test_query_llm_response = input("\nTest response synth on test data: ")
# if len(test_query_llm_response) > 0:
#     response = response_synthesizer.synthesize(
#        test_query_llm_response, 
#        nodes=[Node(text="Electricity can be generated from renewable sources such as wind or solar energy")]
#     )

### Optional user verifiation of summaries
#doc_id = "_"
#while len(doc_id) > 0:
#    doc_id = input("\nEnter doc-ID to print doc summary: ")
#    try:
#        summary = doc_summary_index.get_document_summary(doc_id)
#        print(summary)
#    except Exception as e:
#        print("GET DOC SUMMARY FAILED " + str(e))


# The retriever will retrieve a set of relevant nodes for a given index.
# Optional user verifiation of retreival matching
# retreival_match = "_"
# while len(retreival_match) > 0:
#     retreival_match = input("\nEnter string to test retreival: ")
#     try:
#         retrieved_nodes = retriever.retrieve(retreival_match)
#         try:
#             print("retrieved_nodes: " + str(len(retrieved_nodes)))
#             print("score: " + str(retrieved_nodes[0].score))
#             print("text: " + retrieved_nodes[0].node.get_text()) 
#         except Exception as e:
#             print("PRINTING RETREIVED NODES FAILED: " + str(e))
#     except Exception as e:
#         print(f"An exception occurred: {e}")
#         traceback.print_exc()
#         stack_trace = traceback.format_exc()
#         print(stack_trace)


# 
# Example Nodes as a knowledge base
# from llama_index.schema import Node
# test_nodes = [
#     Node(text="The Earth revolves around the Sun."),
#     Node(text="The Moon orbits the Earth."),
#     Node(text="Gravity is the force of attraction between two bodies."),
#     Node(text="Photosynthesis is the process by which green plants make their own food."),
#     Node(text="Electricity can be generated from renewable sources such as wind or solar energy.")
#     # ... you can add more nodes as needed
# ]# 