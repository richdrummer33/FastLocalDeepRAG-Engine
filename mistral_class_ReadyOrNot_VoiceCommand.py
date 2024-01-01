# [https://python.langchain.com/docs/integrations/providers/ctransformers] from langchain.llms import CTransformers 
from ctransformers import AutoConfig, Config, AutoModelForCausalLM, AutoTokenizer
import transformers
from transformers import pipeline
import winsound
import time
import socket

# TODO NEW INSTALLS:
#   [https://local-llm-function-calling.readthedocs.io/en/latest/generation.html#llama-cpp]
#   pip install local-llm-function-calling[llama-cpp]

# TODO:
#   - Whisper assistant launches this script when the window, the active window is ready or not and calls commands when that's the active window.
#   - Implement RAG for getting commands from a doc, based on (split) voice commands - prevents spanning the large language model with a full command list every prompt.
#   - Talkback to the user with sfx if the command is or is not recognized.
#   - Fine tune the model behavior in HuggingFace website.


class NotificationType:
    WARNING = "C:\Windows\Media\Windows Exclamation.wav"
    SUCCESS = "success.wav"

# SwatGameVoiceCommandExecutor is a class that uses the Mistral AI model to parse a given (transcribed) voice command into sequence of keystrokes (instead of the player manually clicking and keystroking) to execute commands for the game Ready or Not
class SwatGameVoiceCommandExecutor:
    default_instruction = """Please convert the following plain text input into a list of comma-separated values representing keystroke commands based on these command options:

    DOOR commands [MAIN MENU]:
        [1] Stack Up (see STACK UP [SUB MENU 1] below)
        [2] Open (see OPEN [SUB MENU 2] below)

    STACK UP commands [SUB MENU 1]:
        [1] Split
        [2] Left
        [3] Right
        [4] Auto
    
    OPEN commands [SUB MENU 2]:
        [1] Clear
        [2] Clear with Flashbang
        [3] Clear with Stinger
        [4] Clear with CS Gas
        [5] Clear with Launcher
        [6] Clear with Leader"""

    # Scan Sub-menu:
    #     [1] Slide
    #     [2] Pie
    #     [3] Peek

    def load_model(self):
        print("\033[95m\nLoading Model...\n\033[0m")
        self.conf = AutoConfig(Config(temperature=0, repetition_penalty=1.1, 
                         batch_size=52, max_new_tokens=4096, 
                         context_length=4096, gpu_layers=50,
                         stream=False))
        # model_type="mistral"
        # mistral-7b-instruct-v0.1.Q4_K_M.gguf"
        # CONSIDER [https://python.langchain.com/docs/integrations/providers/ctransformers]
        #   llm = Ctranformers(AutoModelForCausalLM.from_pretrained("D:\\Data\\LLM-models\\models\\orca-2-7b.Q5_K_M.gguf", 
        self.llm = AutoModelForCausalLM.from_pretrained("D:\\Data\\LLM-models\\models\\orca-2-7b.Q5_K_M.gguf", 
                                           model_type="llama", config=self.conf)

        self.play_notification_sound(NotificationType.SUCCESS)
        print("\033[92m\nModel Loaded!\n\033[0m")


    def load_pipeline(self):
        print("\033[95m\nLoading Pipeline...\n\033[0m")
        
        tokenizer = AutoTokenizer.from_pretrained("D:\\Data\\LLM-models\\models\\orca-2-7b.Q5_K_M.gguf")

        pipe = pipeline(
            model=self.llm,
            tokenizer=tokenizer,
            task="text-generation",
            return_full_text=True, 
            temperature=0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=64,  # mex number of tokens to generate in the output
            repetition_penalty=1.1  # without this output begins repeating
        )

        print(pipe("AI is going to", max_new_tokens=35))
        pipe("AI is going to", max_new_tokens=35, do_sample=True, temperature=0, repetition_penalty=1.1)

    
    def __init__(self):
        self.client = None
        self.sent_first_message = False
        # [NOTE] Prompt refinement: https://chat.openai.com/share/0184113f-6c94-4ea5-876f-dfc410d141cd

        self.load_model()
        self.load_pipeline()



    # ********************************************************************************************************************
    #
    #  TODO: Use functionary to determine the mode to use (e.g. write code, format text, etc.)
    #        https://github.com/MeetKai/functionary
    #
    #  OG Mistral prompts:
    #       f"[INST] {self.default_instruction} {prompt} [/INST]" 
    #       f"<s>[INST] {self.default_instruction} {prompt} [/INST]" 
    #
    # ********************************************************************************************************************
    def generate_output(self, user_message, callback = None):
        
        
        instruction = self.default_instruction
        # if the 1st word of the user message contains the keyword, then we are in code mode. not equals, but contains
        if self.do_code in user_message.lower().split()[0]:
            print("\033[95m\nGenerating code...\n\033[0m")
            instruction = self.code_instruction
        
        #if(self.sent_first_message): 
        mistral_prompt = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant"
        # else: mistral_prompt = f"<|im_start|>system\n{self.default_instruction}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant"
        
        # print the prompt to the console
        print(f"\033[95m\nPrompt:{mistral_prompt}\n\033[0m")
        print("\033[95m\nGenerating output from prompt...\n\033[0m")
        self.sent_first_message = True        
        
        full_response = ""
        for answer in self.llm(mistral_prompt, temperature=0.2, repetition_penalty=1.1, 
                         batch_size=52, max_new_tokens=4096, stream=True):
            print(answer, end="", flush=True) # stream out to console
            full_response += answer
        
        if callback is not None:
            callback()

        return full_response


    def play_notification_sound(self, notification_type):
        if notification_type == NotificationType.WARNING:
            sound_path = NotificationType.WARNING
        elif notification_type == NotificationType.SUCCESS:
            sound_path = NotificationType.SUCCESS
        winsound.PlaySound(sound_path, winsound.SND_FILENAME)


    def run_socket_client(self):
        
        if self.client is None:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.connect(('localhost', 12345))  # Connect to the server
            while not self.client:
                time.sleep(5)
                client.connect(('localhost', 12345))

        #  await messages from the server
        data = self.client.recv(1024)
        if not data:
            self.run_socket_client()
        
        text_to_llm = data.decode()
        # callback is this function
        response = self.generate_output(text_to_llm, self.run_socket_client)
        client.close()

if __name__ == "__main__":
    # init chatbot
    chatbot = MistralChatbot()
    # start socket client
    chatbot.run_socket_client()
    