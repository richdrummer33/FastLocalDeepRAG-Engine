# WORKS!!!
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


print("\033[95m\nLoading Model...\n\033[0m")
conf = AutoConfig(Config(temperature=0.7, repetition_penalty=1.1, 
                         batch_size=52, max_new_tokens=4096, 
                         context_length=4096, gpu_layers=50,
                         stream=True))

# mistral-7b-instruct-v0.1.Q4_K_M.gguf"
# model_type="mistral"
llm = AutoModelForCausalLM.from_pretrained("./models/orca-2-7b.Q5_K_M.gguf", 
                                           model_type="llama", config = conf)
play_notification_sound(NotificationType.SUCCESS)
print("\033[92m\nModel Loaded!\n\033[0m")

#####################
# Consider batch mode for multiple prompts
# https://python.langchain.com/docs/integrations/llms/huggingface_pipelines
#
# Consider callbacks!
# https://python.langchain.com/docs/integrations/llms/huggingface_textgen_inference
#
# Explore chatbot with memory
#####################
system_message = "You are a helpful assistant"
while True:
    prompt = input("\nEnter prompt: ")

    # f"<s>[INST] {prompt} [/INST]"
    mistral_prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant"

    print("\033[95m\nGenerating output from promt...\n\033[0m")

    for answer in llm(mistral_prompt, temperature=0.7, repetition_penalty=1.1, 
                         batch_size=52, max_new_tokens=2048, stream=True):
        print(answer, end="", flush=True)
