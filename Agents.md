## Autogen Agents and Function calling  

# 1. "user_proxy", UserProxyAgent

User Proxy is used to take users questions and direct them to `askStrategyAgent` or `askResearchAgent`  based on type of query


I have created instance of user ProxyAgent which acts as the user proxy for selecting between agents 

We have got 2 instances of Agents Research_agent and Strategy_agent, the proxy agent can dynamically sect the agents for performing agent specific tasks

# 2. "assistant_for_Research", AssistantAgent

The `AssistantAgent` named `assistant_for_user_research` is configured to act as a helpful assistant within a multi-agent system. Here's a breakdown of what each part of the code is doing:

- `name`: This assigns the name "assistant_for_user_research" to the agent, which is used to identify it within the system.

- `system_message`: This is a message that sets the context for the agent's role. The message "You are a helpful assistant. Reply TERMINATE when the task is done" instructs the agent to continue assisting until it receives a "TERMINATE" command, indicating the end of its task.

- `function_map`: This is a dictionary that maps the string "askResearchAgent" to the function `askResearchAgent`. This mapping allows the agent to call the `askResearchAgent` function when needed. The `askResearchAgent` function is likely defined elsewhere in the code and is responsible for handling research-related queries.

- `llm_config`: This is a configuration dictionary for the agent's language model (LLM) settings. It includes several parameters:
  - `timeout`: Sets the maximum time (in seconds) the agent will wait for a response from the LLM before timing out, which is set to 600 seconds.
  - `cache_seed`: A seed value for caching purposes, set to 42, which can help with reproducibility of results.
  - `config_list`: A list of configurations for the LLM, which is referenced but not defined within this snippet. It likely contains details such as the model type and API keys.
  - `temperature`: Sets the creativity level of the LLM's responses, with 0 indicating deterministic (less random) responses.
  - `functions`: An array of function definitions that the agent can use. In this case, there is one function defined:
    - `name`: "askResearchAgent" is the name of the function.
    - `description`: Provides a description of the function, indicating that it should be used to ask the `askResearchAgent` a question.
    - `parameters`: Defines the expected parameters for the function. It expects an object with a "message" property, which is a string containing the question to be asked.
    - `required`: Specifies which parameters are required, in this case, the "message".

This agent is part of a larger system that includes other agents and functions. 

The `askResearchAgent` function is particularly important as it seems to be the mechanism through which the agent can perform research tasks by querying the Web . 

The agent's behavior, including how it interacts with the LLM and utilizes the `askResearchAgent` function, is determined by the configuration provided in `llm_config`.

# 3. "assistant_for_Strategy", AssistantAgent
The`AssistantAgent` named `assistant_for_user_strategy`  is specifically designed to assist with strategy-related queries. 

Here's a detailed breakdown of its configuration:

- `name`: This sets the agent's name to "assistant_for_user_strategy", which is used to identify it within the system.

- `system_message`: This message, "You are a helpful assistant. Reply TERMINATE when the task is done", serves as an instruction to the agent, indicating that it should continue to assist until it receives a "TERMINATE" command, signaling the end of its task.

- `function_map`: This is a dictionary that maps the string "askStrategyAgent" to the function `askStrategyAgent`. This mapping allows the agent to call the `askStrategyAgent` function when needed. The `askStrategyAgent` function is likely defined elsewhere in the code and is responsible for handling strategy-related queries.

- `llm_config`: This is a configuration dictionary for the agent's language model (LLM) settings, containing several parameters:
  - `timeout`: Sets the maximum time (in seconds) the agent will wait for a response from the LLM before timing out, which is set to 600 seconds.
  - `cache_seed`: A seed value for caching purposes, set to 42, which can help with reproducibility of results.
  - `config_list`: A list of configurations for the LLM, which is referenced but not defined within this snippet. It likely contains details such as the model type and API keys.
  - `temperature`: Sets the creativity level of the LLM's responses, with 0 indicating deterministic (less random) responses.
  - `functions`: An array of function definitions that the agent can use. In this case, there is one function defined:
    - `name`: "askStrategyAgent" is the name of the function.
    - `description`: Provides a description of the function, indicating that it should be used to ask the `askStrategyAgent` a question.
    - `parameters`: Defines the expected parameters for the function. It expects an object with a "message" property, which is a string containing the question to be asked.
    - `required`: Specifies which parameters are required, in this case, the "message".

This agent is part of a larger system that includes other agents and functions. 

The `askStrategyAgent` function is particularly important as it is the mechanism through which the agent can perform strategy-related tasks by querying a strategy database. 

The agent's behavior, including how it interacts with the LLM and utilizes the `askStrategyAgent` function, is determined by the configuration provided in `llm_config`.

# 4. "askResearchAgent" Function 

The `askResearchAgent` function is a complex function calling that integrates several components to perform a research task using a retrieval-augmented approach. Here's a step-by-step breakdown of what each part of the function is doing:

1. **GoogleQuery Search**:
   - A `GoogleQuery` object is created with an API key (`SERP_API_KEY`).
   - The `perform_search` method of the `GoogleQuery` object is called with the user's message as the query. This method uses the SerpApi service to perform a Google search and retrieve the results.
   - The `extract_insights` method processes the search results to extract useful information, such as titles, snippets, and links, and synthesizes them into a summary.
   - The insights are then saved to a file named 'google.txt' using the `save_contents_to_file` function.

2. **Text Splitting**:
   - An instance of `RecursiveCharacterTextSplitter` is created with separators specified as newline, carriage return, and tab characters. This text splitter is used to divide text into smaller chunks for storage in a database.

3. **Embedding Function**:
   - An `OpenAIEmbeddingFunction` object is created using an API key (`OPENAI_API_KEY`) and a model name ("text-embedding-ada-002"). This function is responsible for creating embeddings of text chunks, which are vector representations that capture the semantic meaning of the text.

4. **Retrieval-Augmented User Proxy Agent (ResearchAgent)**:
   - A `RetrieveUserProxyAgent` named "ResearchAgent" is instantiated. This agent is configured to never ask for human input (`human_input_mode="NEVER"`).
   - The `retrieve_config` dictionary specifies the task as "qa" (question-answering), the path to the document containing the insights ('google.txt'), the custom text splitting function (`recur_spliter.split_text`), and the embedding function (`openai_ef`). It also indicates that the agent should create the retrieval system if it does not already exist (`get_or_create=True`).

5. **Retrieval-Augmented Assistant Agent**:
   - A `RetrieveAssistantAgent` named "research assistant" is created. This agent is configured with a system message and a language model configuration (`llm_config`) that includes a timeout, cache seed, and a list of configurations (`config_list`).

6. **Initiating Chat and Retrieving Response**:
   - The `ResearchAgent` initiates a chat with the `assistant` by sending the user's message as a problem.
   - The chat messages between the `ResearchAgent` and the `assistant` are accessed, and the content of the second message (index 1) is retrieved. This message likely contains the response generated by the `assistant` based on the question and the context provided by the `ResearchAgent`.
   - The `ResearchAgent` is reset, which likely clears its state for the next interaction.

7. **Return Response**:
   - The function returns the content of the response message, which is the result of the retrieval-augmented question-answering process.

This function demonstrates a sophisticated integration of search API, text splitting, embedding generation, and retrieval-augmented conversational agents to answer research questions. The process involves searching for relevant information, processing and storing that information, and then using it to generate an informed response to the user's query.

# "askStrategyAgent" Function

The `askStrategyAgent` function is designed to handle strategy-related queries by leveraging a retrieval-augmented approach. Here's a breakdown of the function's workflow:

1. **Embedding Function Initialization**:
   - An instance of `OpenAIEmbeddingFunction` is created using the `embedding_functions` module. This function will generate embeddings for text using OpenAI's API, with the API key obtained from the environment variable `OPENAI_API_KEY` and using the model "text-embedding-ada-002". Embeddings are vector representations of text that capture semantic meaning and are used for similarity comparisons.

2. **Text Splitting**:
   - A `RecursiveCharacterTextSplitter` instance is created with specified separators (newline, carriage return, and tab characters). This splitter will be used to divide text into smaller chunks, which is useful for processing large documents or for storing in a database.

3. **StrategistAgent Creation**:
   - A `RetrieveUserProxyAgent` named "StrategistAgent" is instantiated. This agent is configured to never require human input (`human_input_mode="NEVER"`).
   - The `retrieve_config` dictionary specifies the task as "qa" (question-answering), the path to the document containing relevant information ('./extracted.txt'), the custom text splitting function (`recur_spliter.split_text`), and the embedding function (`openai_ef`). The `get_or_create` parameter indicates that the agent should create the retrieval system if it does not already exist.

4. **Retrieval-Augmented Assistant Agent Creation**:
   - A `RetrieveAssistantAgent` named "strategy assistant" is created. This agent is configured with a system message and a language model configuration (`llm_config`) that includes a timeout, cache seed, and a list of configurations (`config_list`).

5. **Initiating Chat and Retrieving Response**:
   - The `StrategistAgent` initiates a chat with the `assistant` by sending the user's message as a problem.
   - The chat messages between the `StrategistAgent` and the `assistant` are accessed, and the content of the second message (index 1) is retrieved. This message likely contains the response generated by the `assistant` based on the question and the context provided by the `StrategistAgent`.
   - The `StrategistAgent` is reset, which likely clears its state for the next interaction.

6. **Return Response**:
   - The function returns the content of the response message, which is the result of the retrieval-augmented question-answering process.

In summary, the `askStrategyAgent` function uses a retrieval-augmented system to answer strategy-related questions. 

It searches a document for relevant information, splits the text into manageable chunks, and uses embeddings to find the most relevant sections. These sections are then used by a language model to generate a response to the user's query.

The process is automated and does not require human intervention, as indicated by the `human_input_mode="NEVER"` configuration.

