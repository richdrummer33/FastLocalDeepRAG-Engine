# WORKS (no custom llm - just the all-mpnet-base-v2 for query)
# IDEATION: "AI Tools and Project Ideas" in obsidian projet
# GPT CHAT: https://chat.openai.com/share/2633e029-fb49-4734-b132-c5fb013708ba

import os
from sentence_transformers import SentenceTransformer
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from ctransformers import AutoModelForCausalLM, AutoConfig, Config
import winsound
import torch

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

root_directory = "D:/Git/ebook-GPT-translator-refined"
sentences = read_files_recursively(root_directory)
#print(sentences)
model = SentenceTransformer('./sentence-transformers/all-mpnet-base-v2')

# embeddings = model.encode(sentences)
print("\033[95m\nGenerating embeddings...\n\033[0m")
required_exts = [".md", ".py", ".txt", ".json"]
embeddings = SimpleDirectoryReader(
    "D:/Git/ebook-GPT-translator-refined", 
    recursive=True,
    required_exts = required_exts
    ).load_data()

# Indexing with Llama Index
print("\033[95m\nGenerating vector store...\n\033[0m")
index = VectorStoreIndex.from_documents(embeddings)
query_engine = index.as_query_engine()

# Loading and running your causal LM as before
print("\033[95m\nLoading Model...\n\033[0m")
conf = AutoConfig(Config(temperature=0.8, repetition_penalty=1.1,
                         batch_size=52, max_new_tokens=2048,
                         context_length=2048, gpu_layers=50,
                         stream=True))

llm = AutoModelForCausalLM.from_pretrained("./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                                           model_type="mistral", config = conf)
play_notification_sound(NotificationType.SUCCESS)
print("\033[92m\nModel Loaded!\n\033[0m")

while True:
    prompt = input("\nEnter prompt: ")
    mistral_prompt = f"<s>[INST] {prompt} [/INST]"
    
    # Querying the indexed documents
    print("\033[94m\nQuerying indexed documents...\n\033[0m")
    response = query_engine.query(mistral_prompt)
    print("\033[92m\nQuery Response:\n\033[0m", response, "\n")
    
    print("\033[95m\nGenerating output from prompt...\n\033[0m")
    for answer in llm(
        "user query: " + prompt + " relevant data: " + str(response),
        temperature=0.8,
        repetition_penalty=1.1,
        batch_size=52,
        max_new_tokens=2048,
        stream=True
        ):
        print("\033[96m", answer, "\033[0m", end="", flush=True)
    print("\n")  # Adding a line break after the answer for better formatting




