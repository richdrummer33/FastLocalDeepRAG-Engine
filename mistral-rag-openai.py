# WORKS!!!
import winsound
import torch
import openai  # Import the OpenAI library
import os
from langchain.llms import OpenAI
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
import streamlit as st
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    set_global_service_context
)
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

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

#####################################################################
#####################################################################

documents = "D:\Git\EscapeRoom3DGitLab\Assets\Scripts" # "C:/Users/richd/Desktop/test-rag" # "D:/Git/ebook-GPT-translator-refined"

### Setup OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY") # Set your OpenAI API key as an environment variable

#### Fields and Definitions
model_embeddings_path = "./sentence-transformers/all-mpnet-base-v2"
data_path = "D:\\Git\\EscapeRoom3DGitLab\\Assets\\Scripts"

# Make an llm obj
llm = OpenAI(temperature=0.1, verbose=True, model_name="gpt-4")

# Create vectorstore info object - metadata repo?
embed_model = HuggingFaceEmbeddings(model_name=model_embeddings_path)
service_context = ServiceContext.from_defaults(
    chunk_size=2048,
    llm=llm,
    embed_model=embed_model
)
set_global_service_context(service_context) 
store  = VectorStoreIndex.from_documents(documents, service_context=service_context)
vectorstore_info = VectorStoreInfo(
    name="unity_project",
    description="a big ol mp project",
    vectorstore=store
)
# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)


# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)
st.title('🦜🔗 GPT Unity Project Analyser')
# Create a text input box for the user
prompt = st.text_input('Input your prompt here')
play_notification_sound(NotificationType.SUCCESS)

### Ready!
# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    response = agent_executor.run(prompt)
    # ...and write it out to the screen
    st.write(response)

    # With a streamlit expander  
    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt) 
        # Write out the first 
        st.write(search[0][0].page_content) 
