import os
import asyncio
import requests
import time
import json
import openai

from urllib.parse import quote_plus
from openai import OpenAI
from dotenv import load_dotenv
from azure.cognitiveservices.search.websearch import WebSearchClient
from azure.cognitiveservices.search.websearch.models import SafeSearch
from msrest.authentication import CognitiveServicesCredentials

# NOTE: Bing search is free for 1000 queries per month!
# NOTE: OpenAI is *not cheap* particularly using assisistants - just keep in mind!
#       GPT-4 Turbo (gpt-4-1106-preview) is set here, but you can use GPT-3.5 Turbo (gpt-3.5-turbo-1106) which is 10x cheaper

# Implementation Overview:
# - Creates an OpenAI assistant instance set up to call Bing-search and analysis functions (funcs defined in this script)
# - Uses Bing Search API to conduct internet searches based on the processed user query. NOTE: Can easily be swapped to use the Bing Custom Search service (aka resource) instead (which allows you to define what sites it can search, and what sites it can't search)

# Bing API & Service (resource) Setup:
# 1. Sign Up for Azure
#   - If you don't already have one, create a Microsoft Azure account at Azure Portal.
# 2. Go to the Azure Marketplace within the Azure Portal
# 3. Create a new Bing Search resource 
#   - If you wish to limit the search window to certain URLs or domains, then see comments listing 'Bing Custom Search' below.
# 4. Copy the API key from the Azure Portal

# Overview of the Process:
# 1: User Request: 
#   - The user provides a plain-english request for information (e.g. "What are the best stonks to buy right now?").
# 2. Automated Bing Web-Search: 
#   - OpenAI assistant generates a query to search Bing
#   - Implements the Bing Web Search API to conduct internet searches based on the processed user query.
#   - If the user request isn't clear enough to form a Bing-search query from (example query: "sup man?"), then the assistant will (likely) respond seeking more information.
# 3. Automated Search Result Processing: 
#   - Analyzes and processes the search results obtained from Bing, again using the OpenAI assistant instance.
#   - The assistant will then provide a summary of the search results, tailored to answer the user's initial query.
# Result Analysis and Response: Provides a summary or analysis of the search results, tailored to answer the user's initial query.

# NOTE (references and notes)
#  
# Bing web search (Azure) API (Use Azure marketplace and search for "Bing Search" or "Bing Custom Search" - avoid the old Bing API)
#   CURRENT AZURE API REF: https://docs.microsoft.com/en-us/azure/cognitive-services/bing-web-search/quickstarts/python
#   DEPRECATED API REF: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/quickstarts/sdk/web-search-client-library-python
# 
# Tavily web-search for LLMs (alternative option - probably not as good as Bing search)
#   https://docs.tavily.com/docs/tavily-api/python-sdk
#
# OpenAI Tools (assistants):
#   https://platform.openai.com/docs/assistants/tools
#   https://platform.openai.com/docs/assistants/tools/function-calling
#   RECENT: https://medium.com/@nilakashdas/how-to-build-smart-assistants-using-open-ai-apis-ebf9edc42084 
#   https://medium.com/@assafelovic/how-to-build-an-openai-assistant-with-internet-browsing-ee5ad7625661
#   NOTE, share results with community:
#   https://community.openai.com/t/new-assistants-browse-with-bing-ability/479383/12
#   Multi-function assistants:
#   https://dev.to/esponges/build-the-new-openai-assistant-with-function-calling-52f5
# 
# Use a connection to Azure OpenAI on your data.
#   https://learn.microsoft.com/en-us/microsoft-copilot-studio/nlu-generative-answers-azure-openai

# TODO: 
#   Have LLM scrape data from sites linked by the Bing web search, and then analyze the *scraped data* (rather than raw search results) to answer the user's question.
#   Consider using local LLM for generating Bing search query (it creates an optimized keyword search query from user's plain english request) 
#   Consider using local LLM for performing Bing search analysis (for answering user's question)


# Load environment variables
load_dotenv()

# OpenAI API Key
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# NOTE: OLD BING API fields
subscription_key = "79f43664b4a343e38b4017f758ee80f3"
# search_client = WebSearchClient(endpoint="https://api.bing.microsoft.com/", credentials=CognitiveServicesCredentials(subscription_key))

# NOTE: NEW BING API fields (API migrated to azure marketplace)
# custom_config_id = "define this if you are using 'Bing Custom Search' service (aka resource) instead of 'Bing Search'"
searchTerm = "microsoft"
# NOTE: This URL is not the same as the one listed in the Azure resource portal. It has the additional v7.0/search? to specify the resource function.
url = 'https://api.bing.microsoft.com/v7.0/search?' #  + 'q=' + searchTerm + '&' + 'customconfig=' + custom_config_id

# OpenAI Model Configuration
base_model = "gpt-4-1106-preview"
max_tokens = 7000 
temperature = 0.2

u_request = ""
s_query = ""
s_results = ""
run = None

############################################################################################################
### OPENAI FUNCTIONS: Functions to perform a Bing search and process the results
############################################################################################################

# OPENAI FUNCTION: Function to perform a Bing search
def perform_bing_search(user_request):
  global u_request
  global s_query
  global s_results

  u_request = user_request
  print(f"Generating a search_query for bing based on this user request: {user_request}")
  openai_prompt = "Generate a search-engine query to satisfy this user's request: " + user_request
  response = client.chat.completions.create(
      model=base_model,
      messages=[{"role": "user", "content": openai_prompt}],
  )
  # Get the response from OpenAI
  bing_query = response.model_dump_json(indent=2)
  s_query = bing_query
  print(f"Bing search query: {bing_query}. Now executing the search...")
  
  bing_response = run_bing_search(user_request)
  s_results = bing_response
  return bing_response

# OPENAI FUNCTION: Function to process Bing search results
def process_search_results(search_results):
  global u_request
  global s_query
  global s_results

  print(f"Analyzing/processing Bing search results")

  # Use GPT to analyze the Bing search results
  prompt = f"Analyze these Bing search results: '{s_results}'\nbased on this user request: {u_request}"
  
  response = client.chat.completions.create(
      model=base_model,
      messages=[{"role": "user", "content": prompt}],
  )
  analysis =  response.choices[0].message.content.strip()

  print(f"Analysis: {analysis}")
  # Return the analysis
  return analysis


############################################################################################################
### ANALYSIS: Perform a Bing search and process the results
############################################################################################################

def run_bing_search(search_query):
  # Returns data of type SearchResponse 
  # https://learn.microsoft.com/en-us/python/api/azure-cognitiveservices-search-websearch/azure.cognitiveservices.search.websearch.models.searchresponse?view=azure-python
  try:
    base_url = "https://api.bing.microsoft.com/v7.0/search?"
    encoded_query = quote_plus(search_query)
    bing_search_query = base_url + 'q=' + encoded_query # + '&' + 'customconfig=' + custom_config_id --> uncomment this if you are using 'Bing Custom Search'
    r = requests.get(bing_search_query, headers={'Ocp-Apim-Subscription-Key': subscription_key})
  except Exception as err:
    print("Encountered exception. {}".format(err))
    raise err
  
  # Old API
  #try:
  #  web_data = search_client.web.search(query=search_query)
  #except Exception as err:
  #  print("Encountered exception. {}".format(err))
  #  raise err
  
  response_data = json.loads(r.text)
  results_text = ""
  for result in response_data.get("webPages", {}).get("value", []):
    results_text += result["name"] + "\n"
    results_text += result["url"] + "\n"
    results_text += result["snippet"] + "\n\n"
    print(f"Title: {result['name']}")
    print(f"URL: {result['url']}")
    print(f"Snippet: {result['snippet']}\n")

  return results_text

############################################################################################################
### OPENAI ASSISTANT RUN MANAGEMENT
############################################################################################################

# Function to wait for a run to complete
def wait_for_run_completion(thread_id, run_id):
    while True:
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        print(f"Current run status: {run.status}")
        if run.status in ['completed', 'failed', 'requires_action']:
            return run

# Function to handle tool output submission
def submit_tool_outputs(thread_id, run_id, tools_to_call, run, tool_output_array=None, func_override=None):
    global s_results
    print(f"Submitting tool outputs for thread_id: {thread_id}, run_id: {run_id}, tools_to_call: {tools_to_call}")
    if tool_output_array == None:
      tool_output_array = []
    for tool in tools_to_call:
        output = None
        tool_call_id = tool.id
        function_name = func_override if func_override else tool.function.name
        function_args = tool.function.arguments

        if function_name == "perform_bing_search":
            print("[function call] perform_bing_search...")
            output = perform_bing_search(user_request = json.loads(function_args)["user_request"])

        elif function_name == "process_search_results":
            print("[function call] process_search_results...")
            output = process_search_results(json.loads(function_args)["search_results"]) #search_results = s_results) #json.loads(function_args)["search_results"]) #(search_results = s_results) 

        if output:
          print("[function result] Appending tool output array...")
          tool_output_array.append({"tool_call_id": tool_call_id, "output": output})

    return client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id,
        run_id=run_id,
        tool_outputs=tool_output_array
    )

# Function to print messages from a thread
def print_messages_from_thread(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    message = ""
    print("\n====== Assistant Response ======\n")
    for msg in messages:
      if msg.role == "assistant":
        print(f"{msg.role}: {msg.content[0].text.value}")
        message += f"{msg.role}: {msg.content[0].text.value}\n"
    
    return message

# Initialize the assistant and its features and tools
assistant = client.beta.assistants.create(
  instructions="""You are a Q&A bot who performs web searches to respond to user queries. 
  Call function 'perform_bing_search' when provided a user query. 
  Call function 'process_search_results' if the search results contain the info needed to answer the user query.
  Call function 'perform_bing_search' again if the search results do not contain the info needed to answer the user query.
  """,
  model=base_model,
  tools=[
    {
      "type": "code_interpreter"
    },
    {
    "type": "function",
    "function": {
      "name": "perform_bing_search", # Function itself should run a GPT OpenAI-query that asks the OpenAI to generate (and return) a Bing-search-query.
      "description": "Determine a Bing search query from the user_request for specified information and execute the search",
      "parameters": {
        "type": "object",
        "properties": {
          "user_request": {"type": "string", "description": "The user's request, used to formulate a Bing search message"},
        },
        "required": ["user_request"]
      }
    }
  }, 
  {
    "type": "function",
    "function": {
      "name": "process_search_results", # Function itself should send the Bing seardh results to openai to assess the results, and then return the results of that assessment to the user.
      "description": "Analyze Bing search results and return a summary of the results that most effectively answer the user's request",
      "parameters": {
        "type": "object",
        "properties": {
          "search_results": {"type": "string", "description": "The results from the Bing search to analyze"},
        },
        "required": ["search_results"]
      }
    } 
  }
]
)
assistant_id = assistant.id
print(f"Assistant ID: {assistant_id}")


# Create a thread
thread = client.beta.threads.create()
print(f"Thread: {thread}")

# Ongoing conversation loop
while True:
    prompt = input("\nYour request: ")
    if prompt.lower() == 'exit':
        break

    status = "na"
    
    #while status != "completed":
      # Create a message and run
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt,
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,   
    )
    print(f"Run ID: {run.id}")
    # Wait for run to complete
    
    run = wait_for_run_completion(thread.id, run.id)
    while run.status == 'requires_action':
        print("Run requires action 1")
        run = submit_tool_outputs(thread.id, run.id, run.required_action.submit_tool_outputs.tool_calls, run) # **error on this line**
        run = wait_for_run_completion(thread.id, run.id)
        time.sleep(1)
    if run.status == 'failed':
        print(run.error)
        continue
    # Print messages from the thread
    #prompt = print_messages_from_thread(thread.id)
    print_messages_from_thread(thread.id)
    time.sleep(1)