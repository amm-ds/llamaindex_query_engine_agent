import os
import pickle
from typing import Dict, List
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import JSONalyzeQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.prompts import PromptTemplate


class JSONalyzeTool:
    def __init__(self, config: Dict, llm: OpenAI, json_file: str) -> None:
        """
        Initializes the JSONalyzeTool for processing and analyzing JSON data using a specified language model.

        This class sets up a query engine tool that can analyze JSON data and answer questions about it.
        It uses a configuration dictionary to set up the tool and a language model to generate responses.

        Args:
            config (Dict): Configuration settings including prompts and directory paths for JSON files.
            llm (OpenAI): An instance of the OpenAI language model for generating responses.
            json_file (str): The name of the JSON file to be analyzed (without extension).

        Attributes:
            config (Dict): Stored configuration settings.
            llm (OpenAI): The language model instance.
            json_file (str): The name of the JSON file being analyzed.
            tool (QueryEngineTool): The initialized query engine tool for JSON analysis.
        """
        self.config = config
        self.llm = llm
        self.json_file = json_file
        self.tool = self._initialize_jsonalyze_tool()

    def _initialize_jsonalyze_tool(self) -> QueryEngineTool:
        """
        Initializes the JSONalyze tool by creating a query engine for processing the specified JSON data.

        This method performs the following steps:
        1. Loads the prompt template for the specific JSON file.
        2. Reads the JSON data from a pickle file.
        3. Creates a JSONalyzeQueryEngine with the loaded data and configurations.
        4. Wraps the query engine in a QueryEngineTool with appropriate metadata.

        Returns:
            QueryEngineTool: An instance that encapsulates the JSON query engine and its metadata.

        Raises:
            FileNotFoundError: If the specified JSON pickle file is not found.
            KeyError: If required configuration keys are missing.
        """
        # Load the prompt template for this specific JSON file
        prompt_template_str = self.config["prompts"][f"json_tool_{self.json_file}"][
            "prompt"
        ]
        prompt_template = PromptTemplate(prompt_template_str)

        # Construct the path to the pickle file and load the JSON data
        pickle_path = os.path.join(
            self.config["directories"]["json_dir"], f"{self.json_file}.pkl"
        )

        try:
            with open(pickle_path, "rb") as file:
                json_data = pickle.load(file)
            # Note: The JSON data is loaded in pickle format as List[dict],
            # where each dict represents a row of data in the SQLite table to query.
            # This format is required for the JSONalyzeQueryEngine.
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON pickle file not found: {pickle_path}")

        # Initialize the JSONalyzeQueryEngine with the loaded data and configurations
        jsonalyze_engine = JSONalyzeQueryEngine(
            list_of_dict=json_data,
            llm=self.llm,
            table_name=self.json_file,
            verbose=self.config["json_tool"]["verbose"],
            jsonalyze_prompt=prompt_template,
        )

        # Wrap the query engine in a QueryEngineTool with appropriate metadata
        return QueryEngineTool(
            query_engine=jsonalyze_engine,
            metadata=ToolMetadata(
                name=self.config["prompts"][f"json_tool_{self.json_file}"]["name"],
                description=self.config["prompts"][f"json_tool_{self.json_file}"][
                    "description"
                ],
            ),
        )

    def _generate_field_descriptions(self, sample_data: Dict) -> str:
        """
        Generates field descriptions based on the structure of the JSON data.

        This method creates a string of field descriptions, including the field name,
        data type, and a placeholder for a more detailed description.

        Args:
            sample_data (Dict): A sample of the JSON data structure.

        Returns:
            str: A string containing descriptions of the JSON fields, one per line.

        Example:
            Input: {"name": "John", "age": 30}
            Output: "name: (str) Description of name.\nage: (int) Description of age."
        """
        descriptions: List[str] = []
        for key, value in sample_data.items():
            data_type = type(value).__name__
            description = f"{key}: ({data_type}) Description of {key}."
            descriptions.append(description)
        return "\n".join(descriptions)

    def query(self, question: str) -> str:
        """
        Queries the JSON data using the provided question.

        This method sends the question to the JSONalyzeQueryEngine and returns its response.

        Args:
            question (str): The question to be answered based on the JSON data.

        Returns:
            str: The answer generated by the query engine.

        Note:
            The quality and relevance of the answer depend on the underlying JSON data
            and the capabilities of the language model used.
        """
        return self.tool.query_engine.query(question).response
