from ctransformers import AutoModelForCausalLM, AutoConfig, Config
import winsound
import time
import socket

class NotificationType:
    WARNING = "C:\Windows\Media\Windows Exclamation.wav"
    SUCCESS = "success.wav"

class MistralChatbot:

    # keywords 
    do_code = "computer"
    code_instruction = "You are a helpful assistant. If I ask you explicitly to write code, then just give me the script without commentary, but you can put comments in the code where you feel it necessarry."
    default_instruction = "Format the message/text as a user would when typing it as an email or a slack message - clearly and concisely."
    

    def __init__(self):
        self.client = None
        self.sent_first_message = False
        # [NOTE] Prompt refinement: https://chat.openai.com/share/0184113f-6c94-4ea5-876f-dfc410d141cd
        # """
        # You are an AI assistant tasked with processing speech-to-text VOICE TRANSCRIPTS. Your role is twofold:
        #     1. If the transcript *STARTS* with the keyword: "Computer": Perform whatever action is requested in the transcript.
        #     2. If the transcript *DOES NOT* start with the keyword: "Computer": Reformat and adjust the transcript's writing for clarity and context relevance. This includes adding carriage returns, punctuation, bullet points, and correcting any misinterpreted words, as seems fit.
        # Do not respond with commentary or queries/clarification. *Only respond with the requested code or formatted transcript*.
        # """

        #"""You are to correct any issus you see in this speech-to-text transcription. 
#
        #                            Common issues include:
        #                            - Improper punctuation
        #                                e.g. (requires correction) PROVIDED TRANSCRIPT: "Hi john, I'm doing the dishes... right now and I'm going to go to the store later.", but the '...' does not fit. This would be an appropriate correction: "Hi John, I'm doing the dishes right now, and I'm going to go to the store later."
        #                                e.g. (does not require correction) Transcription: "Hi John... I'm doing the dishes right now, and I'm going to go to the store later.", and the ... implies a thoughtful pause
        #                            - Contextual punctuation, like a period where an excaimation point should be for emphasis
        #                                e.g. Transcription: "Holy crap that is wild.", but it's contextually beneficial to say "Holy crap that is wild!"
        #                            - Incorrectly interpreted words
        #                                e.g. Transcription: "I ran across the lawn and jumped into the pile of heaves", but it's far more likely that the speaker said "I ran across the lawn and jumped into the pile of leaves"
#
        #                            If you see issues, respond with the corrected transcription.
        #                            Only respond with the corrected transcription - no commentary.
        #                            If you see no issues, respond with "No issues detected!".
        #                            """

        print("\033[95m\nLoading Model...\n\033[0m")
        self.conf = AutoConfig(Config(temperature=0.2, repetition_penalty=1.1, 
                         batch_size=52, max_new_tokens=4096, 
                         context_length=4096, gpu_layers=50,
                         stream=False))
        # mistral-7b-instruct-v0.1.Q4_K_M.gguf"
        # model_type="mistral"
        self.llm = AutoModelForCausalLM.from_pretrained("D:\\Data\\LLM-models\\models\\orca-2-7b.Q5_K_M.gguf",
                                           model_type="llama", config=self.conf)

        self.play_notification_sound(NotificationType.SUCCESS)
        print("\033[92m\nModel Loaded!\n\033[0m")

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
                self.client.connect(('localhost', 12345))

        #  await messages from the server
        data = self.client.recv(1024)
        if not data:
            self.run_socket_client()
        
        text_to_llm = data.decode()
        # callback is this function
        response = self.generate_output(text_to_llm, self.run_socket_client)
        self.client.close()

if __name__ == "__main__":
    # init chatbot
    chatbot = MistralChatbot()
    # start socket client
    chatbot.run_socket_client()
    