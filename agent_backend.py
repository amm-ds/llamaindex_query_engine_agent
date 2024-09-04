import os
from typing import Tuple, List, Optional
from llama_index.core import Settings
from llama_index.core.memory import (
    SimpleComposableMemory,
    ChatSummaryMemoryBuffer,
    VectorMemory,
)
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from configs.config import get_config
from tools.vector_tool import VectorTool
from tools.json_tool import JSONalyzeTool


class QueryEngineAgent:
    """
    A versatile AI Assistant backend system utilizing LlamaIndex components.

    This class orchestrates the setup and operation of a sophisticated query processing system,
    integrating various tools, memory components, and language models to provide
    intelligent responses to user queries.

    Attributes:
        config (dict): Configuration settings for the agent.
        llm (OpenAI): The main language model for generating responses.
        embed_model (HuggingFaceEmbedding): The embedding model for vector representations.
        vector_tool (VectorTool): Tool for vector-based information retrieval.
        jsonalyze_tools (List[JSONalyzeTool]): Tools for analyzing JSON data.
        tools (List[Union[VectorTool, JSONalyzeTool]]): Combined list of all available tools.
        composable_memory (SimpleComposableMemory): Memory system for context retention.
        system_prompt (str): The system prompt guiding the agent's behavior.
        agent_runner (AgentRunner): The main agent runner for processing queries.
    """

    def __init__(self):
        """
        Initialize the QueryEngineAgent with all necessary components and settings.

        This constructor sets up the entire backend system, including:
        - Loading configurations
        - Initializing language and embedding models
        - Setting up global settings
        - Preparing tools for information retrieval and analysis
        - Configuring memory systems
        - Creating the agent runner
        """
        self.config = get_config()
        self.llm, self.embed_model = self._initialize_components()
        self._setup_settings()
        self.vector_tool = self._initialize_vector_tool()
        self.jsonalyze_tools = self._initialize_jsonalyze_tools()
        self.tools = [self.vector_tool] + self.jsonalyze_tools
        self.composable_memory = self._initialize_memory()
        self.system_prompt = self.config["prompts"]["system_prompt"]
        self.agent_runner = self._create_agent_runner()

    def _setup_settings(self) -> None:
        """
        Configure global settings for LlamaIndex components.

        This method sets up the global Settings class with the initialized language model,
        embedding model, and chunking parameters for consistent use across the system.
        """
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = self.config["chunking"]["chunk_size"]
        Settings.chunk_overlap = self.config["chunking"]["chunk_overlap"]

    def _initialize_components(self) -> Tuple[OpenAI, HuggingFaceEmbedding]:
        """
        Initialize the core NLP components: the language model and embedding model.

        Returns:
            Tuple[OpenAI, HuggingFaceEmbedding]: The initialized LLM and embedding model.
        """
        # Initialize the main language model (LLM)
        llm = OpenAI(
            model=self.config["models"]["llm"],
            temperature=self.config["llm_params"]["temperature"],
        )

        # Initialize the embedding model
        embed_model = HuggingFaceEmbedding(
            model_name=self.config["models"]["embedding"],
            trust_remote_code=self.config["embedding_params"]["trust_remote_code"],
        )
        return llm, embed_model

    def _initialize_vector_tool(self) -> VectorTool:
        """
        Initialize the vector-based information retrieval tool.

        This tool is used for efficient similarity search and information extraction
        from vector representations of data.

        Returns:
            VectorTool: An initialized vector tool for information retrieval.
        """
        return VectorTool(
            config=self.config, llm=self.llm, embed_model=self.embed_model
        )

    def _initialize_jsonalyze_tools(self) -> List[JSONalyzeTool]:
        """
        Initialize tools for analyzing and querying JSON-structured data.

        This method creates a list of JSONalyzeTool instances, each configured
        to work with a specific JSON file as defined in the configuration.

        Returns:
            List[JSONalyzeTool]: A list of initialized JSON analysis tools.
        """
        return [
            JSONalyzeTool(config=self.config, llm=self.llm, json_file=json_file)
            for json_file in self.config["json_tool"]["files"]
        ]

    def _initialize_memory(self) -> SimpleComposableMemory:
        """
        Initialize the memory system for context retention and retrieval.

        This method sets up a composable memory system that includes:
        1. A chat summary memory buffer for maintaining conversation history.
        2. A vector memory for efficient retrieval of relevant past information.

        Returns:
            SimpleComposableMemory: An initialized composable memory system.
        """
        # Initialize chat history
        chat_history = []

        # Set up the summarizer LLM for the chat summary memory
        summarizer_llm = OpenAI(
            model=self.config["memory"]["summarizer_llm"],
            max_tokens=self.config["memory"]["summarizer_max_tokens"],
        )

        # Create the chat summary memory buffer
        chat_summary_memory_buffer = ChatSummaryMemoryBuffer.from_defaults(
            chat_history=chat_history,
            llm=summarizer_llm,
            token_limit=self.config["memory"]["token_limit"],
        )

        # Set up the vector memory for efficient information retrieval
        vector_memory = VectorMemory.from_defaults(
            vector_store=None,  # Will be initialized later if needed
            embed_model=self.embed_model,
            retriever_kwargs=self.config["memory"]["retriever_kwargs"],
        )

        # Combine the memory components into a composable memory system
        return SimpleComposableMemory.from_defaults(
            primary_memory=chat_summary_memory_buffer,
            secondary_memory_sources=[vector_memory],
        )

    def _create_agent_runner(self) -> AgentRunner:
        """
        Create and configure the main agent runner for query processing.

        This method sets up a FunctionCallingAgentWorker with the initialized tools,
        system prompt, and other configuration parameters. It then wraps this worker
        in an AgentRunner for easier interaction.

        Returns:
            AgentRunner: The configured agent runner ready for processing queries.
        """
        # Create the function-calling agent worker
        agent_worker = FunctionCallingAgentWorker.from_tools(
            llm=self.llm,
            tools=[tool.tool for tool in self.tools],
            system_prompt=self.system_prompt,
            max_function_calls=self.config["agent"]["max_function_calls"],
            verbose=self.config["agent"]["verbose"],
            allow_parallel_tool_calls=False,
        )

        # Determine whether to use memory based on configuration
        memory_param = (
            self.composable_memory
            if self.config["memory"].get("use_memory", False)
            else None
        )

        # Create and return the agent runner
        return agent_worker.as_agent(memory=memory_param)

    def process_question(self, query: str) -> str:
        """
        Process a user query and generate a response using the agent runner.

        This method sends the query to the agent runner, which utilizes the configured
        tools and memory to generate an appropriate response. It also handles memory
        updates if the memory system is enabled.

        Args:
            query (str): The user's input query to be processed.

        Returns:
            str: The generated response from the agent runner, or an error message if processing fails.
        """
        try:
            # Process the query using the agent runner
            response = self.agent_runner.chat(
                message=query, tool_choice=self.config["agent"]["tool_choice"]
            )

            # Update memory if it's enabled
            if self.config["memory"].get("use_memory", False):
                history = self.composable_memory.get()
                self.composable_memory.put(history[-1])

            return response
        except Exception as e:
            return f"Error processing query: {e}"


if __name__ == "__main__":
    # Initialize the QueryEngineAgent
    assistant = QueryEngineAgent()

    # Main interaction loop
    while True:
        query = input("Enter a query for the assistant (type 'exit' to quit): ")
        if query.lower() == "exit":
            print("Exiting the assistant. Goodbye!")
            break
        response = assistant.process_question(query)
        print(response)
