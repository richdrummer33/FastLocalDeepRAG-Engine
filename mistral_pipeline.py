
# NOT WORKING YET
from ctransformers import AutoModelForCausalLM, AutoConfig, Config, AutoTokenizer
from transformers import pipeline
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
conf = AutoConfig(Config(temperature=0.8, repetition_penalty=1.1, 
                         batch_size=52, max_new_tokens=1024, 
                         context_length=2048))

llm = AutoModelForCausalLM.from_pretrained("./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                                           model_type="mistral", gpu_layers=50, config = conf)
tokenizer = AutoTokenizer.from_pretrained("./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")                                        

play_notification_sound(NotificationType.SUCCESS)
print("\033[92m\nModel Loaded!\n\033[0m")

prompt = "Tell me about AI"
prompt_template=f'''<s>[INST] {prompt} [/INST]
'''

print("\033[95m\nGenerating output from promt...\n\033[0m")
pipe = pipeline(
    "text-generation",
    model=llm,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)

print(pipe(prompt_template)[0]['generated_text'])
tok = tokenizer("prompt_template")
tokens = len(tok['input_ids'])
print(f"Number of tokens: {tokens}")