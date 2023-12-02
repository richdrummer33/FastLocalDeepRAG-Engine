import streamlit as st
from ctransformers import AutoModelForCausalLM, AutoConfig, Config
import winsound

class NotificationType:
    WARNING = "C:\\Windows\\Media\\Windows Exclamation.wav"
    SUCCESS = "C:\\Windows\\Media\\Speech On.wav"

def play_notification_sound(notification_type):
    if notification_type == NotificationType.WARNING:
        sound_path = NotificationType.WARNING
    elif notification_type == NotificationType.SUCCESS:
        sound_path = NotificationType.SUCCESS
    winsound.PlaySound(sound_path, winsound.SND_FILENAME)

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model():
    conf = AutoConfig(Config(temperature=0.8, repetition_penalty=1.1,
                             batch_size=52, max_new_tokens=2048,
                             context_length=2048, gpu_layers=50,
                             stream=True))
    model = AutoModelForCausalLM.from_pretrained("./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                                                 model_type="mistral", config=conf)
    play_notification_sound(NotificationType.SUCCESS)
    return model

# Initialize Streamlit app
st.title("Mistral Chatbot")

# Load the model and play a success sound
if 'model' not in st.session_state:
    st.session_state['model'] = load_model()
    st.text("Model Loaded!")

# Function to generate chat responses
def generate_response(prompt):
    mistral_prompt = f"<s>[INST] {prompt} [/INST]"
    responses = st.session_state.model(mistral_prompt, temperature=0.2, repetition_penalty=1.1,
                                       batch_size=52, max_new_tokens=2048, stream=False)
    full_response = ''.join(responses)  # Adjust this line based on the actual format of the output
    return full_response

# Manage chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to handle send action
def handle_send():
    if 'user_prompt' in st.session_state and st.session_state.user_prompt:
        # Generate and display the response
        full_response = generate_response(st.session_state.user_prompt)
        st.session_state.chat_history.append(full_response)
        # Clear the input box after sending the message
        st.session_state.user_prompt = ""

# Create an input box for user prompts
user_prompt = st.text_input("Enter prompt:", key="user_prompt", on_change=handle_send)

# Display the chat history
st.text("Chat History:")
for response in st.session_state.chat_history:
    st.text(response)

# Add a button to clear the chat history
if st.button("Clear Chat History"):
    st.session_state.chat_history.clear()
