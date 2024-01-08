# WORKS!!!
from ctransformers import AutoModelForCausalLM, AutoConfig, Config
# from llama_index.llms import HuggingFaceLLM
import winsound

# import torch
# torch.set_default_device('cuda')

class Notification:
    WARNING = "C:\\Windows\\Media\\Windows Exclamation.wav"
    SUCCESS = "C:\\Windows\\Media\\Speech On.wav"
    
    def play_notification_sound(notification_type):

        if notification_type == notification_type:
            sound_path = notification_type
        elif notification_type == notification_type:
            sound_path = notification_type
        winsound.PlaySound(sound_path, winsound.SND_FILENAME)

# class def for Mistral
class Local_LLM:
    llm = None
    system_message = "A computer hacker who is super focused and in flow-state. Skilled in brevity. Good at following instructions, doesn't ask questions. Replies in less than 1 sentence, often in a few words."

    # constructor - loads model
    def __init__(self):
        print("\033[95m\nLoading Model...\n\033[0m")
        conf = AutoConfig(Config(temperature=0.7, repetition_penalty=1.1, 
                                batch_size=52, max_new_tokens=500, 
                                context_length=500, gpu_layers=50))

        # mistral-7b-instruct-v0.1.Q4_K_M.gguf" # model_type="mistral"
        # with cuda
        self.llm = AutoModelForCausalLM.from_pretrained("./models/orca-2-7b.Q5_K_M.gguf", model_type="llama", config = conf, device_map = 'cuda')
        
        # done!
        # notification = Notification()
        #notification.play_notification_sound(Notification.SUCCESS)
        print("\033[92m\nModel Loaded!\n\033[0m")

    # generate output from prompt, one shot (no streaming)
    def prompt_llm(self, prompt):
        global llm

        mistral_prompt = f"<|im_start|>system\n{self.system_message}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant"
        output = self.llm(mistral_prompt, temperature=0.3, repetition_penalty=1.1,
                                batch_size=52, max_new_tokens=500, stream=False)
        
        print(f"\033[95m\nPrompt:{output}\n\033[0m")
        return output
    
    # with streaming
    def prompt_llm_stream(self, prompt):
        global llm

        mistral_prompt = f"<|im_start|>system\n{self.system_message}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant"

        full_output = ""
        for answer in self.llm(mistral_prompt, temperature=0.3, repetition_penalty=1.1, 
                         batch_size=52, max_new_tokens=500, stream=True):
            print(answer, end="", flush=True) # stream out to console
            full_output += answer
                                
        return full_output

#####################
# Consider batch mode for multiple prompts
# https://python.langchain.com/docs/integrations/llms/huggingface_pipelines
#
# Consider callbacks!
# https://python.langchain.com/docs/integrations/llms/huggingface_textgen_inference
#
# Explore chatbot with memory
#####################

# if main (aka running from command line)
if __name__ == "__main__":
    # create mistral object
    llm_inst = Local_LLM()

    while True:
        prompt = input("\nEnter prompt: ")
        _ = llm_inst.prompt_llm(prompt)
