# [https://python.langchain.com/docs/integrations/providers/ctransformers] from langchain.llms import CTransformers 
# from ctransformers import AutoConfig, Config, AutoModelForCausalLM, AutoTokenizer
# import transformers
# from transformers import pipeline
import winsound
import time
import socket
import autogen
from screen_reader import ScreenReader

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
## 💾 COMMANDS REFERENCE 💾
## https://veterans-gaming.com/semlerpdx-avcs/profiles/commref/ron91.html/
##############################################################################################################

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
    Assistant that converts transcribed speech into tactical video-game instructions, based solely on the instruction options below.
    Assistant performs the video-game instructions that execute the intent of the plain-english command (video game command for NPC SWAT officers)
    
    {game_commands}

    examples:    
            If the transcription is "bang and clear it", that means instructions: "open", then "clear with flashbang"
            If the transcription is "move your asses to that door ya dinguses" likely means instructions: "stack up", then "auto"
            If the transcription is "stack up and launcher it", that means instructions "stack up", "auto", "open", then "clear with launcher" (auto since it's not specified).
    
    Longer sentences *do not* necessarily mean more commands! The user could be long-winded.
    Command sequences can only transition DOWN the heirarchy - to execute an *action* command. Once an *action* command is completed, the next command sequence must start from the top: e.g. ["stack up", "auto", "open", "clear with flashbang"] *is* valid, but ["stack up", "open", where "clear with flashbang"] is *not* valid.
    Most sequences are 2, generally no longer than 4 commands.
    """

game_commands = """

    # NOTE: This sub-menu is only available after selecting the 'Stack Up' command from [MAIN MENU]
    # CONTEXT: Action instructions, used for moving to DOORS
    STACK UP instructions [SUB MENU 1]:
        Split
        Left
        Right
        Auto

    # NOTE: only available after selecting the 'Open' command from [MAIN MENU]
    # CONTEXT: Action instructions, used for interaction with DOORS
    OPEN instructions [SUB MENU 2]:
        Clear
        Clear with Flashbang
        Clear with Stinger
        Clear with CS Gas
        Clear with Launcher
        Clear with Leader

    # NOTE: only available after selecting the 'Breach' command from [MAIN MENU]
    # CONTEXT: Action instructions, used for forceful interaction with DOORS
    BREACH instruction [SUB MENU 3]:
        Kick
        Shotgun
        C2

    # CONTEXT: Action instructions
    SCAN instruction [SUB MENU 4]:
        Slide
        Pie
        Peek

    # CONTEXT: Action instruction
    STANDARD instruction:
        Move To
        Fall In
        Cover

    # CONTEXT: Action instructions
    RESTRAIN & REPORT instructions:
        Restrain
        Move To
        Fall In
        Cover
        Deploy
    """

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
    "seed": 40, # 42
    "config_list": config_list,
    "temperature": 0.3 # 0.2
}

command_interpreter_bot = autogen.AssistantAgent(
    name="Command Interpreter",
    system_message=system_message #  + "\n\n" + game_commands,
    llm_config=llm_config,
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "web"},
    llm_config=llm_config,
    system_message="Ensure the Assistant provides a correct list of commands from that summary - and when correct, execute them. DO NOT suggest executing commands unless you've validated them. Reply TERMINATE once they are executed. The command list is below.\n\n" + game_commands
)



##############################################################################################################
## Commands
##############################################################################################################

from typing import List, Tuple, Optional
from fuzzywuzzy import process

class Command:
    def __init__(self, name, key, sub_commands=None):
        self.name = name
        self.key = key
        self.sub_commands = sub_commands or {}

# This structure is called a "tree" data structure
commands = {
    "door": {
        "stack_up": Command("Stack Up", "1", sub_commands={
            "split": Command("Split", "1"),
            "left": Command("Left", "2"),
            "right": Command("Right", "3"),
            "auto": Command("Auto", "4"),
        }),
        "open": Command("Open", "2", sub_commands={
            "clear": Command("Clear", "1"),
            "clear_with_flashbang": Command("Clear with Flashbang", "2"),
            "clear_with_stinger": Command("Clear with Stinger", "3"),
            "clear_with_cs_gas": Command("Clear with CS Gas", "4"),
            "clear_with_launcher": Command("Clear with Launcher", "5"),
            "clear_with_leader": Command("Clear with Leader", "6"),
        }),
        "breach": Command("Breach", "3", sub_commands={
            "kick": Command("Kick", "1"),
            "shotgun": Command("Shotgun", "2"),
            "c2": Command("C2", "3"),
        }),
        "scan": Command("Scan", "4", sub_commands={
            "slide": Command("Slide", "1"),
            "pie": Command("Pie", "2"),
            "peek": Command("Peek", "3"),
        }),
    },
    "standard": {
        "move_to": Command("Move To", "1"),
        "fall_in": Command("Fall In", "2"),
        "cover": Command("Cover", "3"),
    },
    "restrain_and_report": {
        "restrain": Command("Restrain", "1"),
        "move_to": Command("Move To", "2"),  # Duplicate keys may cause confusion
        "fall_in": Command("Fall In", "3"),  # Duplicate keys may cause confusion
        "cover": Command("Cover", "4"),      # Duplicate keys may cause confusion
        "deploy": Command("Deploy", "5"),
    },
    "collect_evidence": {
        "collect_evidence": Command("Collect Evidence", "1"),
    },
}

class CommandProcessor:
    def __init__(self):
        self.commands = commands
        self.last_command = None

    def validate_commands(self, command_sequence: List[str]) -> Tuple[bool, Optional[str]]:

        # This is the list of valid main commands (heirarchy level 0)
        print(f"?????valid_main_commands?????")
        valid_main_commands = set()
        for category in self.commands.values():
            for cmd in category.values():
                valid_main_commands.add(cmd.name.lower())

        ### Validate command sequence ###
        for command_str in command_sequence:

            # Ensure case-insensitive comparison
            print(f"?????command_str.lower?????")
            command_str = command_str.lower()  

            print(f"?????command_str in valid_main_commands?????")
            # Check if the entire command string is a main command
            if command_str in valid_main_commands:
                self.last_command = command_str

            # If not, it should be a sub-command under the last main command
            elif self.last_command:
                print(f"?????self.last_command?????")  
                sub_commands = self.commands[self.last_command]
                if command_str not in sub_commands:
                    return False, f"The command '{command_str}' is not a valid sub-command after '{self.last_command}'."
                self.last_command = command_str

            # If there was no valid command at all, it's an error
            else:
                print(f"?????command_str?????")
                try:
                    closest_match = self.fuzzy_match_command(command_str, self.last_command)
                    if closest_match:
                        return False, f"The command '{command_str}' is not recognized. Did you mean '{closest_match}'?"
                except:
                    return False, f"The command '{command_str}' is not recognized."
                
                return False, f"The command '{command_str}' is not recognized."

        return True, "Commands executed successfully. COMMANDS: " + str(command_sequence)

    def validate_single_command(self, command_str: str) -> Tuple[bool, Optional[str]]:
        return self.validate_commands([command_str])

    # Find the closest matching command in the menu node and returns it if over a threshold
    def fuzzy_match_command(input_command, last_command):

        # Escape if no last valid command
        if last_command is None:
            return None
        
        # Get the sub-commands of the last main command
        context_commands = commands
        if last_command:
            context_commands = commands.get(last_command, {}).sub_commands

        # Flatten the context command names for fuzzy matching
        print(f"?????context_command_name?????")
        context_command_names = [cmd.name.lower() for cmd in context_commands[last_command].sub_commands.values()]
        print(f"(((((context_command_name)))))))")


        # Use fuzzy matching to find the closest command
        closest_match, score = process.extractOne(input_command.lower(), context_command_names)
        
        # Define a match score threshold
        threshold = 70
        if score >= threshold:
            return closest_match
        
        return None


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

# define return type for list of commands on screen
class AvailableCommands(BaseModel):
    isValid: bool
    commands: List[str]

# Define the function to be called by the user proxy
from IPython import get_ipython

available_commands = None

@user_proxy.register_for_execution(name="check_avail_commands")
@command_interpreter_bot.register_for_llm(name="check_avail_commands", description="What commands are up on screen, ready for execution *if any*.")
def check_avail_commands() -> AvailableCommands:
    
    # finds the commands on screen, matches them to the note in the tree of commands in CommandProcessor, and returns the list of commands
    print("\033[95m\nChecking available commands...\n\033[0m")
    global available_commands
    available_commands = None # reset
    
    #wait for the commands to be available
    while available_commands is None:

        time.sleep(0.33)

        # ocr
        screen_reader = ScreenReader() 
        _, screen_text = screen_reader.read_screen(False, False)
        print("$$$$$$$grabbed screen text")

        if not screen_text: 
            print("\033[91m\nNo screen text found!\n\033[0m")
            available_commands = None
            return False, None

        # check if any of the top level commands are in the screen text
        commands_found = []
        for category in commands.values():
            for cmd in category.values():
                if cmd.name.lower() in screen_text.lower():
                    commands_found.append(cmd.name.lower())
                    print(f"!!!!!!!!!!!!!!! FOUND CMD ON SCREEN: {cmd.name.lower()} !!!!!!!!!!!!!!!")

        if commands_found:
            available_commands = commands_found
            break


    # return the list of commands
    return True, available_commands



# import this with pip install keyboard
import keyboard as kb
import time
prev_commands = []

@user_proxy.register_for_execution(name="execute_command")
@command_interpreter_bot.register_for_llm(name="execute_command", description="Executes a single game command-menu selection.")
def execute_command(command: Annotated[str, "A game command for an in-game NPC command menu."], ) -> CommandValidationResult:
    
    # some buffer
    time.sleep(0.15)

    global prev_commands
    global available_commands
    if available_commands == None:
        check_avail_commands()
    if available_commands == None:
        print("\033[91m\nNo commands available on screen!\n\033[0m")
        return False, f"No commands available in buffer! Check available commands again to refresh the buffer and before executing."

    # remove lead and trailing whitespace, and lower
    command = command.strip().lower()
    print ("\033[95m\nAttempting execute command ",command,"\n\033[0m")

    # else if the command provided isnt in the list of available commands
    if command not in available_commands:
        available_commands_str = ', '.join(available_commands)
        print("\033[91m\nInvalid command: '{command}'.\n\033[0m")
        return False, f"Invalid command: '{command}'. Current available commands are: " + available_commands_str

    commands_list_flattened = []
    for category in commands.values():  # Iterate over the categories
        for cmd in category.values():  # Iterate over the Command objects in each category
            commands_list_flattened.append(cmd)
            commands_list_flattened.extend(cmd.sub_commands.values())  # Add sub-commands if any

    command_lower = command.lower()
    command_obj = next((cmd for cmd in commands_list_flattened if cmd.name.lower() == command_lower), None)
    if not command_obj:
        print(f"\033[91m\nInvalid command: '{command}'.\n\033[0m")
        return False, f"Invalid command: '{command}'."

    keystroke = command_obj.key
    print ("\033[95m\n**!!!!!!!!!!Executing keystroke** ", keystroke, "\n\033[0m")
    kb.press_and_release(keystroke)
    time.sleep(0.15)

    prev_commands.append(command)
    prev_commands_str = ', '.join(prev_commands)
    
    available_commands = None # reset
    return True, "Command executed successfully! This series of commands has now been executed: " + prev_commands_str


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
    global current_time
    current_time = time.time()
    prev_commands = []

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
    







#@user_proxy.register_for_execution(name="execute_commands")
#@command_interpreter_bot.register_for_llm(name="execute_commands", description="Executes game voice commands.")
# def execute_commands(string_commands_sequence: Annotated[List[str], "A list of voice-commands that result in a singular instruction to an NPC(s) in-game."]) -> CommandValidationResult:
#     
#     # log the LLM eval time
#     duration = time.time() - current_time
#     print ("\033[95m\n!!!!!!!!!!Executing keystrokes. Done in [", duration, "] seconds!\n\033[0m")
# 
#     # Flatten the list of Command objects from the commands dict, including sub-commands
#     commands_list_flattened = []
#     for category in commands.values():
#         for cmd in category.values():
#             commands_list_flattened.append(cmd)
#             commands_list_flattened.extend(cmd.sub_commands.values())
# 
#     for input_command in string_commands_sequence:
#         input_command_lower = input_command.lower()
# 
#         command_obj = next((cmd for cmd in commands_list_flattened if cmd.name.lower() == input_command_lower), None)
# 
#         if not command_obj:
#             print("\033[91m\nInvalid command: '{input_command}'.\n\033[0m")
#             return False, f"Invalid command: '{input_command}'."
#         
#         # It's good! Execute the command
#         keystroke = command_obj.key
#         # Execute the keystroke using kb module
#         print ("\033[95m\n**Executing keystroke** ", keystroke, "\n\033[0m")
#         kb.press_and_release(keystroke)
#         time.sleep(0.15)
# 
#     # !!! WE ARE DONE - RESET THE AGENTS !!!
#     user_proxy.reset()
#     command_interpreter_bot.reset()
# 
#     return True, "Commands executed successfully! Please TERMINATE."


#@user_proxy.register_for_execution(name="validate_commands")
#@command_interpreter_bot.register_for_llm(name="validate_commands", description="Validates game voice commands.")
# def validate_commands(commands: Annotated[List[str], "A series of game commands that result in NPC action(s)."]) -> CommandValidationResult:
#     """
#     Validates a list of commands against the defined menu structure,
#     ensuring sub-menu commands follow the corresponding main menu commands.
#     Returns a tuple (isValid, message).
#     """
#     print("\033[95m\nValidating commands...\n\033[0m")
# 
#     command_tree = CommandProcessor()
#     is_valid, message = command_tree.validate_commands(commands)
#     print(f"~~ Validation result: {is_valid}, {message} ~~")
# 
#     if not is_valid:
#         print(f"\033[91m{message}\n\033[0m")
#         return False, message
# 
#     print("\033[92mAll commands are valid!\n\033[0m")
#     return True, "All commands are valid."
# 


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