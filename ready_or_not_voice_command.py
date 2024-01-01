# [https://python.langchain.com/docs/integrations/providers/ctransformers] from langchain.llms import CTransformers 
# from ctransformers import AutoConfig, Config, AutoModelForCausalLM, AutoTokenizer
# import transformers
# from transformers import pipeline
import winsound
import time
import socket
import autogen
from openai import OpenAI

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



##############################################################################################################
## Defs & Misc
##############################################################################################################

current_time = 0

def play_notification_sound(self, notification_type):
    if notification_type == NotificationType.WARNING:
        sound_path = NotificationType.WARNING
    elif notification_type == NotificationType.SUCCESS:
        sound_path = NotificationType.SUCCESS
    winsound.PlaySound(sound_path, winsound.SND_FILENAME)

system_message = """
    Assistant that converts transcribed speech into a list of tactical video-game commands, based solely on the command options below.
    Find the closest matching command sequence in all cases - e.g. "bang and clear" means "clear with flashbang", or "move your asses to that door ya dinguses" means "stack up auto".
    Reply TERMINATE when done.
    """

game_commands = """
    # NOTE: No two of these commands can be executed at once - e.g. you cannot 'stack up' and then 'open' a door in the same command sequence.
    DOOR commands [MAIN MENU]:
        1 Stack Up (INFO: sub-commands in 'STACK UP [SUB MENU 1]', below)
        2 Open (INFO: sub-commands in 'OPEN [SUB MENU 2]', below)

    # NOTE: This sub-menu is only available after selecting the 'Stack Up' command from [MAIN MENU]
    STACK UP commands [SUB MENU 1]:
        1 Split
        2 Left
        3 Right
        4 Auto
    
    # NOTE: This sub-menu is only available after selecting the 'Open' command from [MAIN MENU]
    OPEN commands [SUB MENU 2]:
        1 Clear
        2 Clear with Flashbang
        3 Clear with Stinger
        4 Clear with CS Gas
        5 Clear with Launcher
        6 Clear with Leader"""
    
        # SCAN commands [SUB MENU]:
        #     [1] Slide
        #     [2] Pie
        #     [3] Peek

import os
# Get from env var
openai_api_key = "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" # OpenAI(api_key=os.environ["OPENAI_API_KEY"])

config_list= [
    {
        "model": "gpt-3.5-turbo-1106", # "TheBloke/Orca-2-7B-GGUF/orca-2-7b.Q4_K_S.gguf",# "TheBloke\CodeLlama-7B-Instruct-GGUF\codellama-7b-instruct.Q4_K_S.gguf", # "TheBloke/Orca-2-7B-GGUF/orca-2-7b.Q5_K_M.gguf",
        # "base_url": "http://localhost:1234/v1", #"http://0.0.0.0:8000",
        # "api_type": "open_ai",
        "api_key": openai_api_key # "sk-111111111111111111111111111111111111111111111111" # required, you don´t need to change this
    }
]

llm_config={
    #"request_timeout": 600,
    "seed": 42, # 42
    "config_list": config_list,
    "temperature": 0.2 # 0.2
}

command_interpreter_bot = autogen.AssistantAgent(
    name="Command Interpreter",
    system_message=system_message + "\n\n" + game_commands,
    llm_config=llm_config,
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "web"},
    llm_config=llm_config,
    system_message="Assistant that ensures the command list provided by your agent is correct - and when correct, executes the keystrokes. The command list is below. Reply TERMINATE when done. \n\n" + game_commands
)


##############################################################################################################
## Commands
##############################################################################################################

class CommandNode:
    def __init__(self, name, sub_commands=None, default_sub_command=None):
        self.name = name
        self.sub_commands = sub_commands or {}
        self.default_sub_command = default_sub_command

    def validate(self, command):
        if command in self.sub_commands:
            return True, None
        elif self.default_sub_command:
            return True, f"Default sub-command '{self.default_sub_command}' selected for '{self.name}'."
        return False, f"Unrecognized sub-command '{command}' for '{self.name}'."


class CommandTree:
    def __init__(self):
        self.commands = {
            "door": CommandNode("door", {
                "stack up": CommandNode("stack up", {
                    "split": "1", "left": "2", "right": "3", "auto": "4"
                }, default_sub_command="auto"),
                "open": CommandNode("open", {
                    "clear": "1", "clear with flashbang": "2", "clear with stinger": "3", 
                    "clear with cs gas": "4", "clear with launcher": "5", "clear with leader": "6"
                }, default_sub_command="clear")
            })
        }
        self.last_command = None
    
    def validate_commands(self, command_sequence):
        for command in command_sequence:
            parts = command.lower().split(maxsplit=1)
            main_command, sub_command = parts if len(parts) > 1 else (parts[0], None)

            if main_command in self.commands:
                self.last_command = self.commands[main_command]
            elif self.last_command and sub_command:
                valid, error_message = self.last_command.validate(sub_command)
                if not valid:
                    return False, error_message
            else:
                return False, f"Command '{command}' is out of order or unrecognized."

        return True, "All commands are valid."

##############################################################################################################
## Class
##############################################################################################################

class NotificationType:
    WARNING = "C:\Windows\Media\Windows Exclamation.wav"
    SUCCESS = "success.wav"


# [NOTE] Prompt refinement: https://chat.openai.com/share/0184113f-6c94-4ea5-876f-dfc410d141cd
# SwatGameVoiceCommandExecutor is a class that uses the Mistral AI model to parse a given (transcribed) voice command into sequence of keystrokes (instead of the player manually clicking and keystroking) to execute commands for the game Ready or Not


llm = None

def __init__(self):
    # self.load_model()
    # self.load_pipeline()
    self.sent_first_message = False
    self.client = None


##############################################################################################################
## Functions
##############################################################################################################

# Define a custom return type
from pydantic import BaseModel
from typing import Annotated, List
class CommandValidationResult(BaseModel):
    isValid: bool
    message: str

# Define the function to be called by the user proxy
from IPython import get_ipython
# from typing_extensions import Annotated
last_main_command = None

@user_proxy.register_for_execution(name="validate_commands")
@command_interpreter_bot.register_for_llm(name="validate_commands", description="Validates game voice commands.")
def validate_commands(commands: Annotated[List[str], "A list of game voice commands that result in an NPC action."]) -> CommandValidationResult:
    """
    Validates a list of commands against the defined menu structure,
    ensuring sub-menu commands follow the corresponding main menu commands.
    Returns a tuple (isValid, message).
    """
    print("\033[95m\nValidating commands...\n\033[0m")

    command_tree = CommandTree()
    is_valid, message = command_tree.validate_commands(commands)

    if not is_valid:
        print(f"\033[91m{message}\n\033[0m")
        return False, message

    print("\033[92mAll commands are valid!\n\033[0m")
    return True, "All commands are valid."


# import this with pip install keyboard
import keyboard as kb
import time

@user_proxy.register_for_execution(name="execute_keystrokes")
@command_interpreter_bot.register_for_llm(name="execute_keystrokes", description="Executes game voice commands.")
def execute_keystrokes(string_commands_sequence: Annotated[List[str], "A list of game voice commands that result in an NPC action."]) -> CommandValidationResult:
    print ("\033[95m\n!!!!!!!!!!Executing keystrokes...\n\033[0m")
    duration = time.time() - current_time
    print ("\033[92m\nDone in ", duration, " seconds!\n\033[0m")

    # Command mappings
    main_menu_commands = {"stack up": "1", "open": "2"}
    stack_up_sub_commands = {"split": "1", "left": "2", "right": "3", "auto": "4"}
    open_sub_commands = {"clear": "1", "clear with flashbang": "2", "clear with stinger": "3", 
                         "clear with cs gas": "4", "clear with launcher": "5", "clear with leader": "6"}

    last_main_command = None

    for command in string_commands_sequence:
        command = command.lower()  # Ensure case-insensitivity
        keystroke = None

        if command in main_menu_commands:
            keystroke = main_menu_commands[command]
            last_main_command = command
        elif last_main_command:
            sub_commands = stack_up_sub_commands if last_main_command == "stack up" else open_sub_commands
            if command in sub_commands:
                keystroke = sub_commands[command]

        if keystroke:
            # Execute the keystroke using kb module
            print ("\033[95m\n**Executing keystroke** ", keystroke, "\n\033[0m")
            kb.press_and_release(keystroke)
            time.sleep(0.5)  # Short delay between keystrokes


    # !!! WE ARE DONE - RESET THE AGENTS !!!
    user_proxy.reset()
    command_interpreter_bot.reset()

    return True, "Commands executed successfully! Please TERMINATE."


# function that takes in a list of commands and uses kb to perform the keystrokes
def get_keystrokes(string_commands_sequence):
    print ("\033[95m\n!!!!!!!!!!Getting keystrokes...\n\033[0m")

    # loop through the list of commands and perform the keystrokes
    for command in string_commands_sequence:
        print ("\033[95m\n!!!!!!!!!!Performing keystrokes for command: ", command, "\n\033[0m")
        

# ********************************************************************************************************************
#  TODO: Use functionary to determine the mode to use (e.g. write code, format text, etc.)
#        https://github.com/MeetKai/functionary
#
#  OG Mistral prompts:
#       f"[INST] {self.default_instruction} {prompt} [/INST]" 
#       f"<s>[INST] {self.default_instruction} {prompt} [/INST]" 
# ********************************************************************************************************************
def handle_voice_command(user_message, callback = None):

    # for assessing how long it takes to generate the output
    current_time = time.time()

    # Initiate conversation with autogen
    print("\033[95m\nGenerating output from prompt...\n\033[0m")
    user_proxy.initiate_chat(
        command_interpreter_bot,
        message=user_message
    )



if __name__ == "__main__":

    # init chatbot
    # commander = GameVoiceCommandProcessor()

    # start socket client
    # commander.run_socket_client()

    # loop to take user input and send to handle_voice_command
    while True:
        user_input = input("Enter command: ")
        handle_voice_command(user_input)
    









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
        
    # def validate_commands
    # print ("\033[95m\n!!!!!!!!!!Validating commands...\n\033[0m")
# 
    # # Define command menus
    # door_commands = ["stack up", "open"]
    # stack_up_commands = ["split", "left", "right", "auto"]
    # open_commands = ["clear", "clear with flashbang", "clear with stinger", 
    #                 "clear with cs gas", "clear with launcher", "clear with leader", 
    #                 "flashbang", "stinger", "cs gas", "launcher", "leader", 
    #                 "with flashbang", "with stinger", "with cs gas", "with launcher", "with leader"]
    # 
    # for command in commands:
    #     parts = command.lower().split()
    #     menu_command = parts[0] if parts else ""
    #     sub_command = " ".join(parts[1:])
# 
    #     # Validate and handle main menu command
    #     if menu_command in door_commands:
    #         last_main_command = menu_command
    #         if not sub_command:
    #             # Default sub-command for 'stack up' and 'open'
    #             default_sub_command = "auto" if menu_command == "stack up" else "clear"
    #             print("\033[92m\nDefault sub-command selected.\n\033[0m")
    #             return True, f"Default sub-command '{default_sub_command}' selected for '{menu_command}'."
    #         continue
# 
    #     # Validate sub-menu command based on last main command
    #     if last_main_command == "stack up" and (menu_command in stack_up_commands or not parts):
    #         continue
    #     elif last_main_command == "open" and (menu_command in open_commands or not parts):
    #         continue
    #     else:
    #         print("\033[91m\nInvalid command sequence or unrecognized command: '{command}'.\n\033[0m")
    #         return False, f"Invalid command sequence or unrecognized command: '{command}'."
    #     
    # print("\033[92m\nAll commands are valid!\n\033[0m")
    # return True, "All commands are valid."