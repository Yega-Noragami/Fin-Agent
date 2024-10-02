import os
import json 
from pathlib import Path
os.environ["AUTOGEN_USE_DOCKER"] = "False"

from dotenv import load_dotenv
load_dotenv()

# Autogen Imports 
import autogen 
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

# Langchain important libraries 
from langchain_community.embeddings import OpenAIEmbeddings  
from langchain_community.vectorstores import FAISS 
from langchain_community.llms import OpenAI 

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# PDF extraction 
import fitz 

#for using chroma db embedding with text file 
from chromadb.utils import embedding_functions

# For google search 
from serpapi import GoogleSearch

#Memgpt libraries 
from memgpt.autogen.memgpt_agent import create_memgpt_autogen_agent_from_config

# Google API 
SERP_API_KEY = os.getenv("SERP_API_KEY")

# Config file for Autogen agent
config_list = [
    {
        "model": "gpt-4",
        "api_key": os.getenv("OPENAI_API_KEY"),
    }
]


config_list_memgpt = [
    {
        "model": "gpt-4",
        "preset": "memgpt_chat",
        "model_wrapper": None,
        "model_endpoint_type": "openai",
        "model_endpoint": "https://api.openai.com/v1",
        "context_window": 8192,  # gpt-4 context window
        "openai_key": os.getenv("OPENAI_API_KEY"),
    },
]
llm_config_memgpt = {"config_list": config_list_memgpt, "seed": 42}




def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a given PDF folder.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def read_contents_of_folder(folder_path):
    """
    Reads the contents of a folder containing PDF files and extracts text from each PDF.
    """
    # Use Path from pathlib to get all PDF files in the folder
    pdf_files = Path(folder_path).glob("*.pdf")

    All_Texts = ''
    # Iterate through each PDF file in the folder
    for pdf_file in pdf_files:
        # Extract text from the current PDF file
        text = extract_text_from_pdf(str(pdf_file))
        # Append extracted text to Full answer
        All_Texts+=str(text)

    return All_Texts
def save_contents_to_file(extracted_text,name):
    # Open the file in write mode
    with open(name, "w") as f:
        # Write the string to the file
        f.write(extracted_text)

# Helper Class to Use Google Search 
class GoogleQuery:
    def __init__(self, api_key):
        self.api_key = api_key

    def perform_search(self, query):
        """
        Performs a Google search using the Serp API.
        
        :param query: The search query.
        :return: The search results.
        """
        search_params = {
            "engine": "google",
            "q": query,
            "api_key": self.api_key
        }

        search = GoogleSearch(search_params)
        results = search.get_dict()
        return results

    def extract_insights(self, search_results):
        """
        Extracts insights from the search results.
        
        :param search_results: The search results from the Serp API.
        :return: A synthesized summary of the top search results.
        """
        insights = []
        for result in search_results.get("organic_results", []):
            title = result.get("title")
            snippet = result.get("snippet")
            link = result.get("link")
            insights.append(f"Title: {title}\nSnippet: {snippet}\nLink: {link}\n")
        
        # Combine the insights into a summary
        synthesized_insights = "\n".join(insights)
        return synthesized_insights


def askStrategyAgent(message):
    #Embedding Function 
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-ada-002"
            )
    
    # split the texts into chunks for storing in database 
    recur_spliter = RecursiveCharacterTextSplitter(separators=["\n", "\r", "\t"])

    #Create StrategistAgent for document Querying
    '''
    The Retrieval-Augmented User Proxy retrieves document chunks based on the embedding similarity,
    and sends them along with the question to the Retrieval-Augmented Assistant.
    '''
    StrategistAgent = RetrieveUserProxyAgent(
    name="StrategistAgent",
    human_input_mode="NEVER",
    retrieve_config={
        "task": "qa",
        "docs_path": "./extracted.txt",
        "custom_text_split_function": recur_spliter.split_text,
        "embedding_function": openai_ef,
        "get_or_create": True,
    },
)
    #Create Assistant for retrievval Agent  
    '''
    The Retrieval-Augmented Assistant employs an LLM to generate code or text as answers 
    based on the question and context provided. If the LLM is unable to produce a satisfactory 
    response, it is instructed to reply with “Update Context” to the Retrieval-Augmented User Proxy.
    '''
    assistant = RetrieveAssistantAgent(
    name="strategy assistant",
    system_message="You are a helpful assistant.",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    },
)
    
    
    StrategistAgent.initiate_chat(assistant, problem= message)
    message = StrategistAgent.chat_messages[assistant]
    StrategistAgent.reset()
    return message[1]['content']

def askResearchAgent(message):

    query_agent = GoogleQuery(api_key=SERP_API_KEY)
    search_results = query_agent.perform_search(message)
    insights = query_agent.extract_insights(search_results)
    save_contents_to_file(insights,'google.txt')

    # split the texts into chunks for storing in database 
    recur_spliter = RecursiveCharacterTextSplitter(separators=["\n", "\r", "\t"])

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-ada-002"
            )
    

    '''
    The Retrieval-Augmented User Proxy retrieves document chunks based on the embedding similarity,
    and sends them along with the question to the Retrieval-Augmented Assistant.
    '''
    ResearchAgent = RetrieveUserProxyAgent(
    name="ResearchAgent",
    human_input_mode="NEVER",
    retrieve_config={
        "task": "qa",
        "docs_path": 'google.txt',
        "custom_text_split_function": recur_spliter.split_text,
        "embedding_function": openai_ef,
        "get_or_create": True,
    },
)  
    
    '''
    The Retrieval-Augmented Assistant employs an LLM to generate code or text as answers 
    based on the question and context provided. If the LLM is unable to produce a satisfactory 
    response, it is instructed to reply with “Update Context” to the Retrieval-Augmented User Proxy.
    '''
    assistant = RetrieveAssistantAgent(
    name="research assistant",
    system_message="You are a helpful assistant.",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    },
)
    ResearchAgent.initiate_chat(assistant, problem= message)
    message = ResearchAgent.chat_messages[assistant]
    ResearchAgent.reset()
    return message[1]['content']

MemGPT = False

def main():


    #  Create a local databse for the files 
    '''
    This Acts as our Database which will be replaced with an actual instance of Vector 
    Database for full functionality. 
    Currently we are parsing the documents for text's and graphs and saving them in a local file called 'extracted.txt'
    '''
    pdf_path = './research reports/'
    extracted_text = read_contents_of_folder(pdf_path)
    save_contents_to_file(extracted_text,'extracted.txt')
    #------- Database created -------- 

    # create assistant for userproxy to call rag based systems 
    '''
    RAG Retrieval Augmented Generation mechanism via Function Calling and use the result to generate output

    As RAG is performed with UserProxyAgent we can only have a single instance of userProxyAgent
    to make RAG calls we have to wrap the agents in a function call

    We have created 2 agents called assistant_for_Strategy and assistant_for_Strategy which will call the 
    respective autogen agents 
    '''
    assistant_for_Strategy = autogen.AssistantAgent(
        name = "assistant_for_user_strategy", 
        system_message="You are a helpful assistant . Reply TERMINATE when the task is done", 
        function_map={
            "askStrategyAgent":askStrategyAgent
        },
        llm_config={
            "timeout": 600,
            "cache_seed": 42,
            "config_list": config_list,
            "temperature": 0, 
            "functions":[
                {
                    "name": "askStrategyAgent",
                    "description": "ask StrategyAgent when you are asked to answer a question",
                    "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "question to ask StrategyAgent. Ensure the question includes enough context. The StrategyAgent does not know the conversation between you and the user unless you share the conversation with the StrategyAgent.",
                        },
                    },
                    "required": ["message"],
                },
                },
            ]
        }
    )

    assistant_for_Research = autogen.AssistantAgent(
    name = "assistant_for_user_research", 
    system_message="You are a helpful assistant . Reply TERMINATE when the task is done", 
    function_map={
            "askResearchAgent": askResearchAgent,
        },
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
        "temperature": 0, 
        "functions":[
            {
                "name": "askResearchAgent",
                "description": "ask askResearchAgent when you are asked to answer a question",
                "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "question to ask askResearchAgent. Ensure the question includes enough context. The askResearchAgent does not know the conversation between you and the user unless you share the conversation with the askResearchAgent.",
                    },
                },
                "required": ["message"],
            },
            },
        ]
    }
)

    '''
    Create instance of user ProxyAgent which acts as the user proxy for selecting between agents 
    We have got 2 instances of Agents Research_agent and Strategy_agent, the proxy agent can dynamically 
    sect the agents for performing agent specific tasks
    '''
    if MemGPT == False:
        user_proxy = autogen.UserProxyAgent(
            name = "User_proxy",
            system_message="You are an helpful agent that direct questions to Research_agent or Strategy_agent and then displays the output returned by the the agents",
            human_input_mode="ALWAYS",
            max_consecutive_auto_reply=10,
        )
    else:
        user_proxy = create_memgpt_autogen_agent_from_config(
            "User_proxy",
            llm_config=llm_config_memgpt,
            system_message="You are an helpful agent that direct questions to Research_agent or Strategy_agent ",
        )

    '''
    We have created an instance of Group Chat where user_proxy agent , Research_agent and Strategy_agent can be called
    '''
    groupchat = autogen.GroupChat(
        agents= [user_proxy, assistant_for_Research,assistant_for_Strategy ],
        messages=[],
        max_round=30,
        # speaker_selection_method="round_robin",
)
    
    manager= autogen.GroupChatManager(
        groupchat=groupchat ,
        llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    }
)
    # Ask for user query 
    query = input('Enter your Question:')
     # 'What's your overall recommendation related to global equity strategies?'

    # Call Autogen Agents.
    user_proxy.initiate_chat(manager, message= query)

     
if __name__ == "__main__":
    main()

