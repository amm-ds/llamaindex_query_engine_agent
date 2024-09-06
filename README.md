# 🦙 LlamaIndex Query Engine Agent 🤖

## 🚀 Introduction

Welcome to the LlamaIndex Query Engine Agent! This project showcases a powerful and flexible framework that leverages LlamaIndex and large language models (LLMs) to create a versatile question-answering system. While this example implementation focuses on the fictional NeoGadget X1 product, the underlying architecture is designed to be easily adaptable to various domains and data sources.

The agent demonstrates the capability to handle complex queries, interact with multiple data formats (including structured and unstructured data), and provide accurate, context-aware responses. The modular design allows for seamless integration of new data sources, making it an ideal starting point for building sophisticated AI-powered information retrieval systems across diverse industries and use cases.

Whether you're looking to create a customer support chatbot, a research assistant, or a domain-specific knowledge base, this LlamaIndex Query Engine Agent provides a robust foundation that can be customized to meet your specific needs. By showcasing best practices in LLM application development, data integration, and query processing, this project serves as both a practical tool and a learning resource for developers exploring the cutting edge of AI-assisted information systems.

## 🔧 Technical Specifications

- **Framework**: LlamaIndex 0.8.x
- **Language**: Python 3.9+
- **Main Libraries**: 
  - LlamaIndex: Core framework for building LLM-powered applications
  - HuggingFace Embeddings: For generating vector representations of text
  - SQLite: Used for storing and querying structured data
  - RAGAS: Evaluation framework for assessing the quality of generated answers
  - OpenAI models: Accessed through LlamaIndex for language understanding and generation

## 📦 Installation Guide

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/llamaindex-query-engine-agent.git
   cd llamaindex-query-engine-agent
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   ```
   export OPENAI_API_KEY=<your_api_key>
   ```

## ⚙️ Configuration

The agent is configured through the `config.py` file. Here's a detailed overview of the main sections:

### Models Configuration
```python
cfg["models"] = {"llm": "gpt-3.5-turbo", "embedding": "BAAI/bge-large-en-v1.5"}
cfg["llm_params"] = {"temperature": 0.3}
cfg["embedding_params"] = {"trust_remote_code": False}
```

- `models`:
  - `llm`: Specifies the LLM to use. "gpt-3.5-turbo" is set as the default for a good balance of performance and cost.
  - `embedding`: Defines the embedding model. "BAAI/bge-large-en-v1.5" is chosen for its effectiveness in semantic search tasks.
- `llm_params`:
  - `temperature`: Controls the randomness of the LLM's outputs. A lower value (0.3) favors more deterministic responses.
- `embedding_params`:
  - `trust_remote_code`: A safety setting for the embedding model. Set to False to prevent execution of untrusted code.

### Directories Configuration
```python
cfg["directories"] = {
    "document_dir": "data/vector",
    "json_dir": "data/json",
    "index_name": "collections/storage",
}
```

This section defines the directory structure for storing various data types:
- `document_dir`: Location for unstructured document data
- `json_dir`: Directory for structured JSON data
- `index_name`: Path for the LlamaIndex storage

### Tools Configuration
```python
cfg["vector_tool"] = {
    "rerank_top_n": 5,
    "content_info": "Detailed information extracted from various documents, including metadata.",
}

cfg["json_tool"] = {
    "verbose": True,
    "files": ["customer_interactions", "product_info"],
}
```

- `vector_tool`: Configures the vector-based retrieval tool:
  - `rerank_top_n`: Number of top results to consider for reranking
  - `content_info`: Description of the content type for context
- `json_tool`: Settings for JSON data processing:
  - `verbose`: Enables detailed logging
  - `files`: List of JSON files to process

### Agent Configuration
```python
cfg["agent"] = {
    "tool_choice": ["vector_tool", "product_info_tool", "customer_interactions_tool"],
    "max_function_calls": 7,
    "verbose": True,
}
```

- `tool_choice`: Specifies which tools the agent can use
- `max_function_calls`: Limits the number of tool calls per query to prevent infinite loops
- `verbose`: Enables detailed logging of the agent's decision-making process

### Memory Settings
```python
cfg["memory"] = {
    "use_memory": True,
    "tokenizer_llm": "gpt-3.5-turbo",
    "summarizer_llm": "gpt-3.5-turbo",
    "retriever_kwargs": {"similarity_top_k": 3},
    "summarizer_max_tokens": None,
    "token_limit": 256,
}
```

- `use_memory`: Enables or disables the agent's memory system. Set to True for a more conversational experience, or False for a task-oriented approach without context retention.
- `tokenizer_llm` and `summarizer_llm`: Specify the models used for tokenization and summarization in memory management.
- `retriever_kwargs`: Configuration for memory retrieval, including the number of similar items to retrieve.
- `summarizer_max_tokens`: Maximum length of memory summaries. None means no limit.
- `token_limit`: Maximum number of tokens to store in memory, helping manage conversation length.

## 🛠️ Tools Explanation

### Vector Tool
The Vector Tool utilizes LlamaIndex's vector store capabilities to retrieve relevant information from unstructured documents about the NeoGadget X1. It includes a reranking step to improve relevance.

Key features:
- Efficient similarity search using vector representations
- Reranking of results for improved accuracy
- Handling of various document formats and metadata

### JSON Tools
Two JSON tools are implemented using SQLite for structured data querying:

1. **Product Info Tool**: 
   - Queries structured data about the NeoGadget X1's specifications, pricing, and availability
   - Enables fast lookup of specific product details

2. **Customer Interactions Tool**: 
   - Retrieves information about customer interactions, support queries, and feedback
   - Facilitates analysis of customer sentiment and common issues

Both tools use SQL queries under the hood for efficient data retrieval and filtering.

## 🧠 Framework Capabilities

The LlamaIndex Query Engine Agent leverages the powerful features of LlamaIndex, including:

1. **Flexible Data Ingestion**: 
   - Easily ingest and process various data types, including unstructured text and structured JSON data
   - Support for multiple file formats and data sources

2. **Advanced Indexing**: 
   - Utilize vector indexing for efficient similarity search and retrieval
   - Hybrid search capabilities combining vector and keyword-based approaches

3. **Query Engines**: 
   - Implement sophisticated query engines that combine vector search with structured data queries
   - Support for complex, multi-step reasoning processes

4. **Tool Use**: 
   - Seamlessly integrate custom tools (like the JSON tools) within the agent's decision-making process
   - Extensible architecture for adding new tools and capabilities

5. **Response Synthesis**: 
   - Generate coherent and context-aware responses by combining information from multiple sources
   - Use of advanced prompting techniques for improved response quality

For more detailed information about LlamaIndex capabilities, check out the [LlamaIndex documentation](https://docs.llamaindex.ai/en/stable/).

## 📊 Evaluation

RAGAS framweork is implemented for comprehensive evaluation of the agent's performance. The evaluation process is implemented in `evaluation.py` and includes the following steps:

1. Loading questions and expected answers from `data/questions.json` (in the example there are 'easy' and 'hard' Q&A)
2. Processing queries through the QueryEngineAgent
3. Evaluating responses using RAGAS metrics:
   - Context Precision
   - Context Recall
   - Faithfulness
   - Answer Relevancy

Additionally, the agent evaluation generates a processed response JSON file that includes the reasoning process of the agent, intermediate responses, and tool usage.

To run the evaluation:

1. Prepare your questions and expected answers in the `data/questions.json` file.
2. Run the evaluation script:
   ```
   python evaluation.py
   ```
3. View the results in:
   - `evaluations/evaluation_file.json`: Overall scores
   - `evaluations/qa_scores.json`: Detailed scores for each question
   - `processed_responses/processed_agent_responses.json`: Processed responses with reasoning and tool usage

The evaluation results provide insights into the agent's performance across different dimensions, helping identify areas for improvement.

## 💬 Interacting with the Agent

The main agent backend is implemented in `agent_backend.py`. It sets up the QueryEngineAgent with all necessary components, including:

- Language and embedding models
- Vector and JSON tools
- Memory system (if enabled)
- Agent runner for processing queries

To start a chat session with the agent:

1. Run the agent script:
   ```
   python agent_backend.py
   ```
2. Enter your questions about the NeoGadget X1 when prompted.
3. The agent will use its tools to retrieve information and provide comprehensive answers.

The agent processes each query through the following steps:
1. Analyzing the query to determine required tools
2. Retrieving relevant information using the selected tools
3. Synthesizing a response based on the retrieved information and conversation context (if memory is enabled)
4. Updating its memory with the new interaction (if memory is enabled)

## 🔮 Future Work

1. **Advanced Agent Strategies**:
   - Implement mixture-of-experts agents for specialized domain knowledge
   - Develop ReAct agents with enhanced reasoning capabilities
   - Create custom reasoning processes for complex, multi-step queries
   - Design multi-task agents capable of handling diverse query types simultaneously

2. **Enhanced Data Modeling and Integration**:
   - Utilize Pydantic for robust data validation and serialization
   - Implement real-time data updating mechanisms for any data source
   - Develop flexible connectors for various external APIs and databases

3. **Sophisticated Query Processing**:
   - Implement multi-hop reasoning for intricate, interconnected queries
   - Develop query decomposition strategies for handling multi-part questions
   - Utilize knowledge graphs for enhanced contextual understanding

4. **Advanced Tool Utilization and Metadata Management**:
   - Implement dynamic tool selection based on query intent and context
   - Develop hybrid search capabilities combining vector, keyword, and semantic search
   - Enhance tool metadata with rich descriptions and usage patterns

5. **Innovative Prompting Strategies**:
   - Implement chain-of-thought prompting for improved reasoning
   - Utilize few-shot learning techniques for adaptable responses
   - Develop dynamic prompt templates that adjust based on query complexity

6. **Expanded Language Model Integration**:
   - Integrate state-of-the-art models like Anthropic's Claude and Groq
   - Implement model switching based on query requirements and performance metrics
   - Develop ensemble methods combining multiple LLMs for enhanced accuracy

7. **Advanced Evaluation and Monitoring**:
   - Implement comprehensive evaluation metrics including faithfulness, coherence, and factual accuracy
   - Utilize multiple LLMs for cross-validation of responses
   - Develop continuous learning and improvement mechanisms based on user feedback and performance metrics

8. **Enhanced User Interaction and Personalization**:
   - Implement advanced conversational memory for context-aware multi-turn interactions
   - Develop user preference learning for tailored responses across various domains
   - Create adaptive interfaces that evolve based on user interaction patterns

9. **Hallucination Mitigation Strategies**:
   - Develop self-checking mechanisms for multi-pass refinement of LLM outputs
   - Enhance Retrieval Augmented Generation (RAG) with specialized prompt engineering
   - Fine-tune smaller LLMs for attribution evaluation to assess response reliability
   - Develop creative fine-tuning approaches on specialized data with expert validation

## 📚 About LlamaIndex

[LlamaIndex](https://docs.llamaindex.ai/en/stable/) is a powerful data framework for LLM-based applications. It provides the following key features:

- 🔍 Flexible data ingestion from various sources
- 📊 Structured data indexing for efficient retrieval
- 🧠 Advanced querying capabilities with multi-step reasoning
- 🔗 Seamless integration with popular LLM providers

LlamaIndex simplifies the process of connecting large language models with external data sources, making it an ideal choice for building robust and scalable AI applications like the NeoGadget X1 information assistant.

Remember, when in doubt, check the [LlamaIndex documentation](https://docs.llamaindex.ai/en/stable/) for detailed information on available features and best practices!

Contributions are welcome from the developer community! If you have ideas for improvements, bug fixes, or new features, please don't hesitate to submit pull requests or open issues on this GitHub repository. Your input is valuable in making this framework even better and more versatile. Let's collaborate to push the boundaries of what's possible with LlamaIndex and AI-powered information retrieval!

Happy coding! 🚀👨‍💻