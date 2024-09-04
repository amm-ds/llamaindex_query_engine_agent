import os
from typing import Dict, List, Callable
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
    Document,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import (
    TitleExtractor,
    KeywordExtractor,
)
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.llms.openai import OpenAI


class VectorTool:
    """
    A tool for managing and querying vector-based document indexes.

    This class provides functionality to create, load, and query a vector index
    of documents. It uses LlamaIndex for document processing, indexing, and retrieval.

    Attributes:
        config (Dict): Configuration settings for the tool.
        llm (OpenAI): Language model for text processing.
        embed_model (HuggingFaceEmbedding): Embedding model for vector representations.
        transformations (List[Callable]): Pipeline of document transformations.
        index (VectorStoreIndex): The vector index of documents.
        tool (QueryEngineTool): Query engine tool for retrieving information.
    """

    def __init__(self, config: Dict, llm: OpenAI, embed_model: HuggingFaceEmbedding):
        """
        Initialize the VectorTool with configuration and models.

        This method sets up the tool by initializing the transformation pipeline,
        building or loading the document index, and creating the query engine tool.

        Args:
            config (Dict): Configuration dictionary containing settings for the tool.
            llm (OpenAI): Language model instance for text processing tasks.
            embed_model (HuggingFaceEmbedding): Embedding model for creating vector representations.
        """
        self.config = config
        self.llm = llm
        self.embed_model = embed_model
        self.transformations = self._setup_transformations()
        self.index = self._build_or_load_index()
        self.tool = self._create_query_engine_tool()

    def _setup_transformations(self) -> List[Callable]:
        """
        Set up the transformation pipeline for document processing.

        This method creates a list of callable transformations that will be applied
        to documents during indexing. The pipeline includes:
        1. Text splitting (SentenceSplitter)
        2. Title extraction (TitleExtractor)
        3. Keyword extraction (KeywordExtractor)
        4. Embedding generation (using the specified embed_model)

        Returns:
            List[Callable]: A list of transformation functions to be applied to documents.
        """
        # Create a sentence splitter with configured chunk size and overlap
        text_splitter = SentenceSplitter(
            chunk_size=self.config["chunking"]["chunk_size"],
            chunk_overlap=self.config["chunking"]["chunk_overlap"],
        )
        # Set up title extractor to process the first 3 nodes
        title_extractor = TitleExtractor(nodes=3, llm=self.llm)
        # Configure keyword extractor to extract 5 keywords
        keyword_extractor = KeywordExtractor(keywords=5, llm=self.llm)

        # Return the list of transformations in the order they should be applied
        return [text_splitter, title_extractor, keyword_extractor, self.embed_model]

    def _build_or_load_index(self) -> VectorStoreIndex:
        """
        Build a new vector index or load an existing one.

        This method checks if a persisted index exists. If not, it creates a new index
        from documents and persists it. Otherwise, it loads the existing index.

        Returns:
            VectorStoreIndex: The loaded or newly created vector index.
        """
        index_path = self.config["directories"]["index_name"]
        if not os.path.exists(index_path):
            # If index doesn't exist, create a new one
            documents = self._load_documents()
            index = VectorStoreIndex.from_documents(
                documents,
                transformations=self.transformations,
                embed_model=self.embed_model,
            )
            # Persist the newly created index
            index.storage_context.persist(persist_dir=index_path)
        else:
            # If index exists, load it from storage
            storage_context = StorageContext.from_defaults(persist_dir=index_path)
            index = load_index_from_storage(storage_context)
        return index

    def _load_documents(self) -> List[Document]:
        """
        Load documents from the specified directory.

        This method uses SimpleDirectoryReader to load all documents from the
        configured document directory.

        Returns:
            List[Document]: A list of loaded Document objects.
        """
        return SimpleDirectoryReader(
            self.config["directories"]["document_dir"]
        ).load_data()

    def _create_query_engine_tool(self) -> QueryEngineTool:
        """
        Create a query engine tool for information retrieval.

        This method sets up a VectorIndexAutoRetriever with the index and configures
        a RetrieverQueryEngine with reranking capabilities.

        Returns:
            QueryEngineTool: A tool that can be used to query the vector index.
        """
        # Set up vector store info for the auto retriever
        vector_store_info = VectorStoreInfo(
            content_info=self.config["vector_tool"]["content_info"],
            metadata_info=[
                MetadataInfo(
                    name=info["name"],
                    type=info["type"],
                    description=info["description"],
                )
                for info in self.config["vector_tool"]["metadata_info"]
            ],
        )

        # Create the auto retriever
        vector_auto_retriever = VectorIndexAutoRetriever(self.index, vector_store_info)
        # Set up reranking postprocessor
        rerank = SentenceTransformerRerank(
            model=self.config["models"]["embedding"],
            top_n=self.config["vector_tool"]["rerank_top_n"],
        )
        # Create the retriever query engine
        retriever_query_engine = RetrieverQueryEngine.from_args(
            vector_auto_retriever, llm=self.llm, node_postprocessors=[rerank]
        )
        # Return the query engine tool with metadata
        return QueryEngineTool(
            query_engine=retriever_query_engine,
            metadata=ToolMetadata(
                name=self.config["prompts"]["vector_tool"]["name"],
                description=self.config["prompts"]["vector_tool"]["description"],
            ),
        )

    def query(self, question: str) -> str:
        """
        Query the vector index using the provided question.

        This method uses the query engine tool to retrieve and generate an answer
        based on the indexed documents.

        Args:
            question (str): The question to be answered based on the vector index.

        Returns:
            str: The answer generated by the query engine.
        """
        return self.tool.query_engine.query(question).response

    def add_documents(self, new_documents: List[Document]) -> None:
        """
        Add new documents to the existing index.

        This method processes new documents, converts them to nodes, and inserts
        them into the existing index. It then persists the updated index.

        Args:
            new_documents (List[Document]): List of new documents to be added to the index.
        """
        # Convert documents to nodes and insert them into the index
        self.index.insert_nodes(
            self.index.node_parser.get_nodes_from_documents(new_documents)
        )
        # Persist the updated index
        self.index.storage_context.persist(
            persist_dir=self.config["directories"]["index_name"]
        )

    def refresh_index(self) -> None:
        """
        Refresh the entire index by reloading all documents and rebuilding the index.

        This method is useful for updating the index when the underlying documents have
        changed significantly. It reloads all documents, rebuilds the index, and
        recreates the query engine tool.
        """
        # Reload all documents
        documents = self._load_documents()
        # Rebuild the index with the latest documents
        self.index = VectorStoreIndex.from_documents(
            documents,
            transformations=self.transformations,
            embed_model=self.embed_model,
        )
        # Persist the newly built index
        self.index.storage_context.persist(
            persist_dir=self.config["directories"]["index_name"]
        )
        # Recreate the query engine tool with the new index
        self.tool = self._create_query_engine_tool()
