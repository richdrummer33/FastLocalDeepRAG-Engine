# WORKS!!!
import winsound
import torch
import time

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# MISC CLASSES
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Example Nodes as a knowledge base
from llama_index.schema import Node
test_nodes = [
    Node(text="The Earth revolves around the Sun."),
    Node(text="The Moon orbits the Earth."),
    Node(text="Gravity is the force of attraction between two bodies."),
    Node(text="Photosynthesis is the process by which green plants make their own food."),
    Node(text="Electricity can be generated from renewable sources such as wind or solar energy.")
    # ... you can add more nodes as needed
]

class NotificationType:
    WARNING = "C:\\Windows\\Media\\Windows Exclamation.wav"
    SUCCESS = "C:\\Windows\\Media\\Speech On.wav"

def play_notification_sound(notification_type):
    if notification_type == NotificationType.WARNING:
        sound_path = NotificationType.WARNING
    elif notification_type == NotificationType.SUCCESS:
        sound_path = NotificationType.SUCCESS
    winsound.PlaySound(sound_path, winsound.SND_FILENAME)

import re
from typing import Tuple, List

import re
from typing import Tuple, List
from itertools import zip_longest

import re
from typing import Tuple, List
from itertools import zip_longest

import streamlit as st

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

retriever = None
response_synthesizer = None

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def init():
    print("\033[95m\nINITIALIZING\n\033[0m")
    #####################
    ### Fields and Definitions
    #####################
    model_path = "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    model_embeddings_path = "./sentence-transformers/all-mpnet-base-v2"

    data_path = "D:\\Git\\twinny-api"  #"D:/Git/Unseen/Assets/Code/" #"D:/Git/EscapeRoom3DGitLab/Assets/Scripts" #"C:/Users/richd/Desktop/test-rag" #"D:/Git/ebook-GPT-translator-refined"
    config_llm_path = "./models/Mistral-7B-Instruct-v0.1/config.json"

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
    print("\033[95m\nLoading Model...\n\033[0m")
    if not use_gpt:
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
    else:
        import openai
        import os
        openai.api_key = os.environ["OPENAI_API_KEY"]
        from llama_index.llms import OpenAI
        llm = OpenAI(temperature=0.1, model="gpt-4")


    ###################################
    ### Embeddings and service context
    ### NOTE FOR LLAMA CPP (GGUF COMPAT): https://gpt-index.readthedocs.io/en/latest/examples/llm/llama_2_llama_cpp.html
    ###################################
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    print("\033[95m\nLoading embeddings model...\n\033[0m")
    embed_model = HuggingFaceEmbeddings(model_name=model_embeddings_path)


    ##############################
    ### Set things up for indexer
    ### NOTE: https://blog.llamaindex.ai/a-new-document-summary-index-for-llm-powered-qa-systems-9a32ece2f9ec
    ##############################
    print("\033[95m\nIndexing...\n\033[0m")
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
    ### Optional user verifiation of summaries
    # test_query_llm_response = input("\nTest response synth on test data: ")
    # if len(test_query_llm_response) > 0:
    #     response = response_synthesizer.synthesize(
    #        test_query_llm_response, 
    #        nodes=[Node(text="Electricity can be generated from renewable sources such as wind or solar energy")]
    #     )


    ###########################
    ### Set up service context
    # Defines the llm and embed models and chunk size to retreive
    # NOTE: Memory-pool issues on docs ingestion: https://github.com/imartinez/privateGPT/issues/181
    #       Should I also increase LLM context window to avoid? Tradeoffs with performance/batching?
    ###########################
    service_context = ServiceContext.from_defaults(
        chunk_size=400, # 1024 # 2048 (I think this too large & caused mem errors on ingest)
        chunk_overlap=40,
        llm=llm,
        embed_model=embed_model
    )
    # set_global_service_context(service_context) # Necessary? Maybe not 

    ###################
    ### Index the docs
    ###################
    from llama_index import VectorStoreIndex, SimpleDirectoryReader, DocumentSummaryIndex, StorageContext
    from llama_index.indices.loading import load_index_from_storage

    build_new_index = input("\033[95m\nPress R to reimport index from storage...\n\033[0m")
    if(build_new_index.lower() != "r" or build_new_index == None):
# ***************************************************************************
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#****************************************************************************
# TODO: FIX LOADING ERROR
#   - Error: "Value Error () shapes () not aligned"
#   - Occurs when loading index from storage
#
# SOLUTION: https://github.com/run-llama/llama_index/issues/819
# ****************************************************************************
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ****************************************************************************
        try:
            storage_context = StorageContext.from_defaults(persist_dir="index")
            doc_summary_index = load_index_from_storage(storage_context)
            print("\033[95m\nLoaded index from storage...\n\033[0m")
        except:
            print("\033[95m\nFailed to load index from storage...\n\033[0m")
            build_new_index = None
    else:
        print("\033[95m\nFetch documents...\n\033[0m")
        reader  = SimpleDirectoryReader(data_path, recursive=True, exclude=['*.meta', '*.preset', '*.bnk', '*.wem', '*.fbx', '*.obj', '*.wav', '*.onnx', '*.otf', '*.mat', '*.png', '*.prefab', '*.unity']) # '*.txt', '.json']
        documents = reader.load_data()
        print("\033[95m\nIndexing documents data...\n\033[0m")
        doc_summary_index = DocumentSummaryIndex.from_documents(
            documents,
            service_context=service_context,
            summary_query=SUMMARY_QUERY,
            response_synthesizer=response_synthesizer, # will use response synth to generate llm response to retreived chunks
            show_progress=True
        )
        doc_summary_index.storage_context.persist("index")  

    ### Optional user verifiation of summaries
    #doc_id = "_"
    #while len(doc_id) > 0:
    #    doc_id = input("\nEnter doc-ID to print doc summary: ")
    #    try:
    #        summary = doc_summary_index.get_document_summary(doc_id)
    #        print(summary)
    #    except Exception as e:
    #        print("GET DOC SUMMARY FAILED " + str(e))

    ##########################
    ### Set up docs retreiver
    ### REF: https://gpt-index.readthedocs.io/en/latest/examples/index_structs/doc_summary/DocSummary.html
    ##########################
    from llama_index.indices.document_summary import DocumentSummaryIndexLLMRetriever, DocumentSummaryIndexEmbeddingRetriever
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


    ###################################
    ### Conversational memory
    ### NOTE NOTE NOTE NOTE : NOT IMPLEMENTED, UNTESTED, UNVERIFIED, UNKNOWN
    ###################################
    #from langchain.agents import Tool
    from llama_index.memory import ChatMemoryBuffer
    from langchain.chains.conversation.memory import ConversationBufferWindowMemory
    # Define tools
    #tools = [
    #    Tool(
    #        name = "LlamaIndex",
    #        func=lambda q: str(index.as_query_engine().query(q)),
    #        description="You are a person who saerches codebases and answers questions related to that codebase in a conversational manner.",
    #        return_direct=True
    #    ),
    #]
    conversational_memory = ConversationBufferWindowMemory(memory_key='chat_history', k=5, return_messages=True)# Initialize agent with conversational memory
    #agent_executor = initialize_agent(tools, llm, agent="conversational-react-description", memory=conversational_memory)


    ###########################
    ### Set up query engine...
    ### Response/summarization mode can include auto-iterative prompt refinement
    ### refine, compact, tree_summarize, etc 
    ### REF (docs retreival): https://gpt-index.readthedocs.io/en/latest/examples/index_structs/doc_summary/DocSummary.html
    ###########################
    from llama_index.query_engine import RetrieverQueryEngine
    print("\033[95m\nQuery engine...\n\033[0m")
    query_engine = RetrieverQueryEngine(    #query_engine = doc_summary_index.as_query_engine(response_mode=ResponseMode.TREE_SUMMARIZE, use_async=True)
        retriever=retriever,
        response_synthesizer=response_synthesizer
        #, memory=conversational_memory,
    ) 
    play_notification_sound(NotificationType.SUCCESS)

    # return the loaded model to the stremlit session state
    return query_engine



##################
### Prompt time!
##################
import streamlit as st

# Initialize Streamlit app
st.title("Mistral Docs Smart-Search")

# Custom CSS to style the chat bubbles
st.markdown("""
    <style>
    .chat-bubble {
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .user-bubble {
        background-color: #E0E0E0;
    }
    .bot-bubble {
        background-color: #F0F0F0;
    }
    </style>
    """, unsafe_allow_html=True)

# Check if the model is already loaded in the session state
if 'model' not in st.session_state:
    print("*RUNNING INIT*")
    st.session_state['model'] = init()
else:
    print("Model already loaded")

# Manage chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
else:
    print("Chat history already loaded")

# Function to generate chat responses using the query_engine
def generate_response(prompt):
    # using cached query engine
    response = st.session_state.model.query(prompt)
    return response

# Function to handle send action
def handle_send():
    user_prompt = st.session_state.user_prompt
    if user_prompt:
        # Generate and display the response
        full_response = generate_response(user_prompt)
        st.session_state.chat_history.append((user_prompt, full_response))
        # Clear the input box after sending the message
        st.session_state.user_prompt = ""

# Create an input box for user prompts
user_prompt = st.text_input("Enter prompt:", key="user_prompt")
st.button("Send", on_click=handle_send)

# Display the chat history with alternating shades
st.text("Chat History:")
for i, (user_input, response) in enumerate(st.session_state.chat_history):
    bubble_class = "user-bubble" if i % 2 == 0 else "bot-bubble"
    st.markdown(f"<div class='chat-bubble {bubble_class}'>You: {user_input}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-bubble {bubble_class}'>Mistral: {response}</div>", unsafe_allow_html=True)

# Add a button to clear the chat history
if st.button("Clear Chat History"):
    st.session_state.chat_history.clear()


#from typing import Dict, List
#
## Use streamlit session state to store values across reruns
#if 'prev_prompts' not in st.session_state:
#    st.session_state['prev_prompts'] = []
#if 'current_index' not in st.session_state:
#    st.session_state['current_index'] = -1
#
#st.title("LLM Search Engine (document summary RAG)")
#
#prompt = st.text_input("Enter prompt:")
#
#if prompt:
#    if "list_ids" in prompt:
#        st.markdown("### Listing IDs...")
#        # Display the list of IDs. Use st.write() or similar functions
#        list_all_doc_key_values(doc_summary_index)
#    elif "doc_id" in prompt:
#        st.markdown("### Querying docstore...")
#        doc_id = prompt.split("doc_id:")[1].split("\n")[0].strip()
#        doc_summary = doc_summary_index.get_document_summary(doc_id)
#        st.write(doc_summary)
#    else:
#        st.markdown("### Generating output from prompt...")
#        response = query_engine.query(prompt)
#
#        try:
#            response.print_response_stream()
#            st.write(response)
#        except Exception as e:
#            st.write("Cannot stream: ", str(e))
#            st.write("Response: ", str(response))
#
        