# [https://python.langchain.com/docs/integrations/providers/ctransformers] from langchain.llms import CTransformers 
# from ctransformers import AutoConfig, Config, AutoModelForCausalLM, AutoTokenizer
# import transformers
# from transformers import pipeline
import winsound
import time
import socket
import autogen

# TODO NEW INSTALLS:
#   [https://local-llm-function-calling.readthedocs.io/en/latest/generation.html#llama-cpp]
#   pip install local-llm-function-calling[llama-cpp]

# TODO:
#   - KEY NOTE: 
#       FUNCTION CALLING -> [https://github.com/abetlen/llama-cpp-python#function-calling] [https://semaphoreci.com/blog/localai]
#       Download Llama-cpp model abetlen/functionary-7b-v1-GGUF for 'abetlen/llama-cpp-python'
#   - Whisper assistant launches this script when the window, the active window is ready or not and calls commands when that's the active window.
#   - Implement RAG for getting commands from a doc, based on (split) voice commands - prevents spanning the large language model with a full command list every prompt.
#   - Talkback to the user with sfx if the command is or is not recognized.
#   - Pipeline for ocr / image recognition from game text [https://huggingface.co/docs/transformers/task_summary].
#   - Fine tune the model behavior in HuggingFace website for my specific use case.
#   ###########################################################################################################
#   - Consider batch GPU inference [https://python.langchain.com/docs/integrations/llms/huggingface_pipelines]
#   ###########################################################################################################

class NotificationType:
    WARNING = "C:\Windows\Media\Windows Exclamation.wav"
    SUCCESS = "success.wav"

# [NOTE] Prompt refinement: https://chat.openai.com/share/0184113f-6c94-4ea5-876f-dfc410d141cd
# SwatGameVoiceCommandExecutor is a class that uses the Mistral AI model to parse a given (transcribed) voice command into sequence of keystrokes (instead of the player manually clicking and keystroking) to execute commands for the game Ready or Not
class GameVoiceCommandProcessor:


##############################################################################################################
## Defines
##############################################################################################################

    llm = None

    system_message = """
    Assistant that converts transcribed speech into a list of tactical video-game commands, based solely on the command options below.
    This assistant first validates the command sequence:
        1. Check commands for validity. If the command sequence is invalid, try again.
        2. Execute the valid list of commands.

    DOOR commands [MAIN MENU]:
        [1] Stack Up (INFO: sub-commands in 'STACK UP [SUB MENU 1]', below)
        [2] Open (INFO: sub-commands in 'OPEN [SUB MENU 2]', below)

    # NOTE: This sub-menu is only available after selecting the 'Stack Up' command from [MAIN MENU]
    STACK UP commands [SUB MENU 1]:
        [1] Split
        [2] Left
        [3] Right
        [4] Auto
    
    # NOTE: This sub-menu is only available after selecting the 'Open' command from [MAIN MENU]
    OPEN commands [SUB MENU 2]:
        [1] Clear
        [2] Clear with Flashbang
        [3] Clear with Stinger
        [4] Clear with CS Gas
        [5] Clear with Launcher
        [6] Clear with Leader"""
    
        # SCAN commands [SUB MENU]:
        #     [1] Slide
        #     [2] Pie
        #     [3] Peek

##############################################################################################################
## Autogen Config
##############################################################################################################
    
    config_list= [
        {
            "model":"TheBloke\Orca-2-7B-GGUF\orca-2-7b.Q4_K_S.gguf", # "TheBloke\CodeLlama-7B-Instruct-GGUF\codellama-7b-instruct.Q4_K_S.gguf", # "TheBloke/Orca-2-7B-GGUF/orca-2-7b.Q5_K_M.gguf",
            "base_url": "http://localhost:1234/v1", #"http://0.0.0.0:8000",
            #"api_type": "open_ai",
        
            "api_key": "sk-111111111111111111111111111111111111111111111111", # required, you don´t need to change this
        }
    ]

    llm_config={
        #"request_timeout": 600,
        "seed": 42, # 42
        "config_list": config_list,
        "temperature": 0 # 0.2
    }

    chatbot = autogen.AssistantAgent(
        name="Command Executor",
        system_message="For game commands, use the provided functions. Reply TERMINATE when done.",
        llm_config=llm_config,
    )

    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config={"work_dir": "game_commands"},
    )


##############################################################################################################
## Functions
##############################################################################################################

    def __init__(self):
        # self.load_model()
        # self.load_pipeline()
        self.sent_first_message = False
        self.client = None
        self.last_main_command = None


    # define functions according to the function description
    from IPython import get_ipython
    from typing_extensions import Annotated

    @user_proxy.register_for_execution()
    @chatbot.register_for_llm(name="validate_command", description="Validates game voice commands.")
    def validate_command(self, commands): # (string list of commands)
        """
        Validates a list of commands against the defined menu structure,
        ensuring sub-menu commands follow the corresponding main menu commands.
        Returns a tuple (isValid, message).
        """
        # Define command menus
        door_commands = ["stack up", "open"]
        stack_up_commands = ["split", "left", "right", "auto"]
        open_commands = ["clear", "clear with flashbang", "clear with stinger", 
                        "clear with cs gas", "clear with launcher", "clear with leader", 
                        "flashbang", "stinger", "cs gas", "launcher", "leader", 
                        "with flashbang", "with stinger", "with cs gas", "with launcher", "with leader"]
        for command in commands:
            parts = command.lower().split()
            menu_command = parts[0] if parts else ""
            sub_command = " ".join(parts[1:])
            # Validate and handle main menu command
            if menu_command in door_commands:
                self.last_main_command = menu_command
                if not sub_command:
                    # Default sub-command for 'stack up' and 'open'
                    default_sub_command = "auto" if menu_command == "stack up" else "clear"
                    return True, f"Default sub-command '{default_sub_command}' selected for '{menu_command}'."
                continue
            # Validate sub-menu command based on last main command
            if self.last_main_command == "stack up" and (menu_command in stack_up_commands or not parts):
                continue
            elif self.last_main_command == "open" and (menu_command in open_commands or not parts):
                continue
            else:
                return False, f"Invalid command sequence or unrecognized command: '{command}'."
        return True, "All commands are valid."

    



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
    def handle_voice_command(self, user_message, callback = None):

        print("\033[95m\nGenerating output from prompt...\n\033[0m")
        
        # Initiate conversation with autogen
        self.user_proxy.initiate_chat(
            self.chatbot,
            message=user_message
        )

        # Process the response
        response = self.user_proxy.wait_for_reply()
        print(response)

        # Optional callback handling
        if callback:
            callback(response)

        return response


    def play_notification_sound(self, notification_type):
        if notification_type == NotificationType.WARNING:
            sound_path = NotificationType.WARNING
        elif notification_type == NotificationType.SUCCESS:
            sound_path = NotificationType.SUCCESS
        winsound.PlaySound(sound_path, winsound.SND_FILENAME)


if __name__ == "__main__":

    # init chatbot
    commander = GameVoiceCommandProcessor()

    # start socket client
    # commander.run_socket_client()

    # loop to take user input and send to handle_voice_command
    while True:
        user_input = input("Enter command: ")
        commander.handle_voice_command(user_input)
    









    ### OLD CODE ###
        
    # USE LM STUDIO INSTEAD!
    # def load_model(self):
    #     print("\033[95m\nLoading Model...\n\033[0m")
    #     self.conf = AutoConfig(Config(temperature=0, repetition_penalty=1.1, 
    #                      batch_size=52, max_new_tokens=4096, 
    #                      context_length=4096, gpu_layers=50,
    #                      stream=False))
    #     
    #     # CONSIDER [https://python.langchain.com/docs/integrations/providers/ctransformers]
    #     # llm = Ctranformers(AutoModelForCausalLM.from_pretrained("D:\\Data\\LLM-models\\models\\orca-2-7b.Q5_K_M.gguf", 
    #     self.llm = AutoModelForCausalLM.from_pretrained("D:\\Data\\LLM-models\\models\\orca-2-7b.Q5_K_M.gguf", 
    #                                        model_type="llama", config=self.conf)
    #
    #     self.play_notification_sound(NotificationType.SUCCESS)
    #     print("\033[92m\nModel Loaded!\n\033[0m")


    # USE LM STUDIO INSTEAD!
    # def load_pipeline(self):
    #     print("\033[95m\nLoading Pipeline...\n\033[0m")
    #     
    #     tokenizer = AutoTokenizer.from_pretrained("D:\\Data\\LLM-models\\models\\orca-2-7b.Q5_K_M.gguf")
    # 
    #     pipe = pipeline(
    #         model=self.llm,
    #         tokenizer=tokenizer,
    #         task="text-generation",
    #         return_full_text=True, 
    #         temperature=0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    #         max_new_tokens=64,  # mex number of tokens to generate in the output
    #         repetition_penalty=1.1  # without this output begins repeating
    #     )
    # 
    #     print(pipe("AI is going to", max_new_tokens=35))
    #     pipe("AI is going to", max_new_tokens=35, do_sample=True, temperature=0, repetition_penalty=1.1)