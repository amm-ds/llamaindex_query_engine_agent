def get_config() -> dict:
    """
    Get the configuration dictionary for the Generic AI Assistant.

    This function initializes and returns a comprehensive configuration dictionary
    that is essential for the operation of the Generic AI Assistant. The configuration
    includes settings for models, directories, prompts, chunking, tools, memory,
    agent behavior, and evaluation parameters.

    Returns:
        dict: The comprehensive configuration dictionary.
    """
    cfg = {}

    # Models
    cfg["models"] = {"llm": "gpt-3.5-turbo", "embedding": "BAAI/bge-large-en-v1.5"}

    cfg["llm_params"] = {"temperature": 0.3}
    cfg["embedding_params"] = {"trust_remote_code": False}

    # Directories
    cfg["directories"] = {
        "document_dir": "data/vector",
        "json_dir": "data/json",
        "index_name": "collections/storage",
    }

    # Prompts
    cfg["prompts"] = {}
    cfg["prompts"][
        "system_prompt"
    ] = """
        You are an advanced AI assistant specializing in providing comprehensive information about the NeoGadget X1 product. 
        Your role is to assist users with queries by leveraging both unstructured and structured data sources.

        Guidelines:
        - Utilize ALL available tools to gather comprehensive information.
        - Analyze and synthesize data from multiple sources to provide accurate and detailed responses.
        - When encountering conflicting information, cross-reference and provide the most up-to-date and reliable answer.
        - If specific information is not available, provide context based on the retrieved data and clearly state any limitations.
        - Ensure responses are informative, helpful, and tailored to the user's query.

        Tool Descriptions:
            vector_tool:
                Retrieves detailed information from various unstructured documents about the NeoGadget X1, including 
                technical specifications, user manuals, and marketing materials.

            product_info_tool:
                Queries a SQLite database to retrieve structured information about the NeoGadget X1 product, including 
                specifications, pricing, and availability.

            customer_interactions_tool:
                Accesses a SQLite database to fetch information about customer interactions related to the NeoGadget X1, including 
                support queries and feedback.

        To use these tools, specify a JSON blob with the following structure:
        {
            "action": $TOOL_NAME,
            "action_input": $INPUT
        }
        
        Each action input produces a $FUNCTION_OUTPUT, that you need to analyze to synthesize the $FINAL_ANSWER.
        You can implement the action loop N times.

        The "action" field must be one of: vector_tool, product_info_tool, customer_interactions_tool
        The "action_input" should contain a relevant query or question related to the user's input. 

        Remember to analyze and combine the outputs from all tool calls to formulate your final response. 
        Ensure that your answer is coherent, accurate, and directly addresses the user's query.
        """

    cfg["prompts"]["vector_tool"] = {
        "name": "vector_tool",
        "description": "Comprehensive information retrieval from various documents about the product",
    }

    cfg["prompts"]["json_tool_product_info"] = {
        "name": "product_info_tool",
        "description": "Retrieves information about the NeoGadget X1 product.",
        "prompt": (
            "You are given a table named: 'product_info' with schema, "
            "generate SQLite SQL query to answer the given question.\n"
            "Table schema:\n"
            "{table_schema}\n"
            "id INTEGER, productName TEXT, modelNumber TEXT, description TEXT, "
            "releaseDate TEXT, price REAL, availableStorage TEXT, colors TEXT, "
            "displaySize REAL, batteryCapacity INTEGER, processor TEXT, ram INTEGER, "
            "cameraSpecs TEXT, warrantyPeriod INTEGER, returnPolicy TEXT, "
            "operatingSystem TEXT, supportEmail TEXT, supportPhone TEXT, "
            "websiteUrl TEXT, inStock INTEGER\n"
            "Here is the structure and description of the fields in the DB:\n"
            "id: (INTEGER) Unique identifier for the product\n"
            "productName: (TEXT) Name of the NeoGadget product\n"
            "modelNumber: (TEXT) Model number of the NeoGadget\n"
            "description: (TEXT) Detailed description of the product\n"
            "releaseDate: (TEXT) Release date of the product (YYYY-MM-DD)\n"
            "price: (REAL) Price of the product in USD\n"
            "availableStorage: (INT) Storage capacity in GB\n"
            "colors: (TEXT) Comma-separated list of available color options\n"
            "displaySize: (REAL) Size of the display in inches\n"
            "batteryCapacity: (INTEGER) Battery capacity in mAh\n"
            "processor: (TEXT) Name of the processor used in the device\n"
            "ram: (INTEGER) Amount of RAM in GB\n"
            "cameraSpecs: (TEXT) Description of the camera specifications\n"
            "warrantyPeriod: (INTEGER) Warranty period in months\n"
            "returnPolicy: (TEXT) Description of the return policy\n"
            "operatingSystem: (TEXT) Name and version of the operating system\n"
            "supportEmail: (TEXT) Email address for customer support\n"
            "supportPhone: (TEXT) Phone number for customer support\n"
            "websiteUrl: (TEXT) URL of the product website\n"
            "inStock: (INTEGER) 1 if the product is in stock, 0 if out of stock\n"
            "Question: {question}\n\n"
            "SQLQuery: "
        ),
    }

    cfg["prompts"]["json_tool_customer_interactions"] = {
        "name": "customer_interactions_tool",
        "description": "Retrieves information about customer interactions for the NeoGadget X1.",
        "prompt": (
            "You are given a table named: 'customer_interactions' with schema, "
            "generate SQLite SQL query to answer the given question.\n"
            "Table schema:\n"
            "{table_schema}\n"
            "id INTEGER, productId INTEGER, customerId INTEGER, conversationId INTEGER, "
            "body TEXT, isIncoming INTEGER, status TEXT, sentDate TEXT, communicationType TEXT\n"
            "Here is the structure and description of the fields in the DB:\n"
            "id: (INTEGER) Unique identifier for the interaction\n"
            "productId: (INTEGER) ID of the product related to the interaction\n"
            "customerId: (INTEGER) ID of the customer involved in the interaction\n"
            "conversationId: (INTEGER) ID of the conversation thread\n"
            "body: (TEXT) Content of the communication message\n"
            "isIncoming: (INTEGER) 1 if the message is from the customer, 0 if from support\n"
            "status: (TEXT) Status of the message (e.g., 'sent', 'received')\n"
            "sentDate: (TEXT) Date and time when the message was sent (YYYY-MM-DD HH:MM:SS)\n"
            "communicationType: (TEXT) Type of communication (e.g., 'email', 'chat')\n"
            "Question: {question}\n\n"
            "SQLQuery: "
        ),
    }

    # Chunking settings
    cfg["chunking"] = {"chunk_size": 128, "chunk_overlap": 4}

    # Vector tool settings
    cfg["vector_tool"] = {
        "rerank_top_n": 5,
        "content_info": "Detailed information extracted from various documents, including metadata.",
        "metadata_info": [
            {
                "name": "document_context",
                "type": "text",
                "description": "Contextual information extracted from documents, including metadata.",
            }
        ],
    }

    # JSON tool settings
    cfg["json_tool"] = {
        "verbose": True,
        "files": [
            "customer_interactions",
            "product_info",
        ],  # Add your JSON file names here
    }

    # Memory settings
    cfg["memory"] = {
        "use_memory": True,
        "tokenizer_llm": "gpt-3.5-turbo",
        "summarizer_llm": "gpt-3.5-turbo",
        "retriever_kwargs": {"similarity_top_k": 3},
        "summarizer_max_tokens": None,
        "token_limit": 256,
    }

    # Agent settings
    cfg["agent"] = {
        "tool_choice": ["vector_tool", "product_info_tool", ""],
        "max_function_calls": 7,
        "verbose": True,
    }

    # Evaluation settings
    cfg["evaluation"] = {
        "questions_file": "data/questions.json",
        "evaluations_output_file": "evaluations/evaluation_file.json",
        "individual_scores_output_file": "evaluations/qa_scores.json",
        "processed_agent_responses_file": "processed_responses/processed_agent_responses.json",
        "evaluation_model": "gpt-3.5-turbo",
        "geval_metrics": [
            {
                "name": "Correctness",
                "criteria": "Correctness - determine if the actual output is correct according to the expected output.",
                "evaluation_params": ["ACTUAL_OUTPUT", "EXPECTED_OUTPUT"],
            },
            {
                "name": "Truthfulness",
                "criteria": "Truthfulness - determine if the actual output is true and informative according to the expected output.",
                "evaluation_params": ["ACTUAL_OUTPUT", "EXPECTED_OUTPUT"],
            },
        ],
    }

    return cfg
