from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Default batch size for MERGE operations
DEFAULT_MERGE_BATCH_SIZE = 32


class FireboltRetriever(BaseRetriever):
    """Simple retriever wrapper for Firebolt vector store.
    
    Provides a convenient interface for retrieving documents from a Firebolt vector store
    with configurable search parameters.
    """
    
    vector_store: "Firebolt"
    search_kwargs: Dict[str, Any] = {}
    
    def __init__(self, vector_store: "Firebolt", search_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any):
        """
        Initialize the retriever.
        
        Args:
            vector_store: The Firebolt vector store instance to use for retrieval.
            search_kwargs: Dictionary of keyword arguments to pass to search methods.
            **kwargs: Additional arguments passed to BaseRetriever.
        """
        if search_kwargs is None:
            search_kwargs = {}
        super().__init__(vector_store=vector_store, search_kwargs=search_kwargs, **kwargs)
    
    def _get_relevant_documents(self, query: str, *, run_manager: Any = None, **kwargs: Any) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The query string to search for.
            run_manager: Optional callback manager.
            **kwargs: Additional keyword arguments (e.g., k for number of results).
        
        Returns:
            List[Document]: List of relevant documents.
        """
        # Merge kwargs with search_kwargs, with kwargs taking precedence
        search_params = {**self.search_kwargs, **kwargs}
        return self.vector_store.similarity_search(query=query, **search_params)
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The query string to search for.
        
        Returns:
            List[Document]: List of relevant documents.
        """
        return self._get_relevant_documents(query)
    
    def get_relevant_documents_with_score(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents with similarity scores for a query.
        
        Args:
            query: The query string to search for.
        
        Returns:
            List[Tuple[Document, float]]: List of (document, score) tuples.
        """
        return self.vector_store.similarity_search_with_score(query=query, **self.search_kwargs)
    
    async def _aget_relevant_documents(self, query: str, *, run_manager: Any = None, **kwargs: Any) -> List[Document]:
        """
        Asynchronously retrieve relevant documents for a query.
        
        Args:
            query: The query string to search for.
            run_manager: Optional callback manager.
            **kwargs: Additional keyword arguments (e.g., k for number of results).
        
        Returns:
            List[Document]: List of relevant documents.
        """
        # Merge kwargs with search_kwargs, with kwargs taking precedence
        search_params = {**self.search_kwargs, **kwargs}
        return await asyncio.to_thread(
            self.vector_store.similarity_search,
            query=query,
            **search_params
        )
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Asynchronously retrieve relevant documents for a query.
        
        Args:
            query: The query string to search for.
        
        Returns:
            List[Document]: List of relevant documents.
        """
        return await self._aget_relevant_documents(query)
    
    async def aget_relevant_documents_with_score(self, query: str) -> List[Tuple[Document, float]]:
        """
        Asynchronously retrieve relevant documents with similarity scores for a query.
        
        Args:
            query: The query string to search for.
        
        Returns:
            List[Tuple[Document, float]]: List of (document, score) tuples.
        """
        return await asyncio.to_thread(
            self.vector_store.similarity_search_with_score,
            query=query,
            **self.search_kwargs
        )


class FireboltSettings(BaseSettings):
    """`Firebolt` client configuration.

    Attribute:
        id (str) : Firebolt client ID to login. Required.
        secret (str) : Firebolt client secret to login. Required.
        engine_name (str) : Firebolt engine name to use. Required.
        database (str) : Database name to find the table. Required.
        account_name (str) : Firebolt account name. Required.
        table (str) : Name of the table that the vector index is built on.
                      Required. Used for add and delete operations.
        index (str, optional) : Index name to operate on. If not specified,
                               will be retrieved from the database by querying
                               information_schema.indexes for vector_search indexes
                               on the table.
        metric (str) : Metric to use for similarity search. 
                       Allowed values: "vector_cosine_ops" (default), "vector_ip_ops", "vector_l2sq_ops".
        llm_location (str, optional) : Location of the LLM API to use for embedding calculation.
                            Required when use_sql_embeddings is True. This should match the name
                            of a LOCATION object created in Firebolt for the LLM service (e.g., Amazon Bedrock).
        embedding_model (str) : Embedding model to use for AI_EMBED_TEXT calls. Required.
                                 Example: "amazon.titan-embed-text-v2:0"
        embedding_dimension (int) : Dimension of the embedding. Defaults to 256.
        batch_size (int) : Batch size for MERGE operations. Defaults to 32.
        api_endpoint (str) : API endpoint to use for the Firebolt client. Defaults to None which uses Firebolt's cloud API endpoint.
        column_map (Dict) : Column type map to project column name onto langchain
                            semantics. Must have keys: `id`, `embedding`, `document`,
                            and `metadata`. The `id` column will be included in search
                            results as metadata['id']. The `metadata` key should be a list of
                            source column names that will be used for metadata at retrieval.
                            For example:
                            .. code-block:: python

                                {
                                    'id': 'id',
                                    'embedding': 'embedding',
                                    'document': 'document',
                                    'metadata': ['file_name', 'page_number']
                                }
                            Defaults to identity map with metadata as empty array.
    """

    id: str = Field(..., json_schema_extra={"env": "FIREBOLT_CLIENT_ID"})
    secret: str = Field(..., json_schema_extra={"env": "FIREBOLT_CLIENT_SECRET"})
    engine_name: str = Field(..., json_schema_extra={"env": "FIREBOLT_ENGINE"})
    database: str = Field(..., json_schema_extra={"env": "FIREBOLT_DB"})
    account_name: str = Field(..., json_schema_extra={"env": "FIREBOLT_ACCOUNT"})
    table: str = Field(..., json_schema_extra={"env": "FIREBOLT_TABLENAME"})
    index: Optional[str] = Field(None, json_schema_extra={"env": "FIREBOLT_INDEX"})
    llm_location: [str] = Field(None, json_schema_extra={"env": "FIREBOLT_LLM_LOCATION"})
    embedding_model: str
    embedding_dimension: int = 256
    batch_size: int = Field(DEFAULT_MERGE_BATCH_SIZE, json_schema_extra={"env": "FIREBOLT_BATCH_SIZE"})
    api_endpoint: Optional[str] = None  # Optional custom API endpoint

    column_map: Dict[str, Union[str, List[str]]] = {
        "id": "id",
        "document": "document",
        "embedding": "embedding",        
        "metadata": []
    }

    metric: str = "vector_cosine_ops"

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="firebolt_",
        extra="ignore",
    )


class Firebolt(VectorStore):
    """Firebolt vector store integration.
    
    NOTE: While this implementation supports similarity_search operations as well as
    add_documents, add_texts, and delete methods, it is recommended to use external tools to populate 
    and manage the Firebolt vector index table through batch operations in order to optimize tablet 
    pruning and reduce the number of tablets.

    Setup:
        Install ``langchain-firebolt``:

        .. code-block:: bash

            pip install langchain-firebolt

    Key init args — client params:
        config: Optional[FireboltSettings]
            Firebolt client configuration.
            
            NOTE: Uses AI_EMBED_TEXT SQL function to calculate embeddings upon insertion and 
            similarity search. This requires the creation of a LOCATION object 
            in Firebolt for the LLM API to use for embedding calculation. The `llm_location` 
            parameter in FireboltSettings must be provided and should match the name of the 
            LOCATION object created in Firebolt (e.g., "llm_api").
    
            See Documentation for more details on creating the LOCATION object: https://docs.firebolt.io/reference-sql/commands/data-definition/create-location-bedrock#create-location-amazon-bedrock
            

    Instantiate:
        .. code-block:: python

            from langchain_firebolt import Firebolt, FireboltSettings

            settings = FireboltSettings(
                id="your_client_id",
                secret="your_client_secret",
                database="my_database",
                table="pdf_documents",
                llm_location="llm_api",  # Required: name of the LOCATION object in Firebolt
                embedding_model="amazon.titan-embed-text-v2:0",  # Required: embedding model
                # index is optional - will be auto-detected if not provided
            )
            vector_store = Firebolt(config=settings)

    Search:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud", k=1)
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

    Search with filter:
        .. code-block:: python

            results = vector_store.similarity_search(
                query="thud", k=1, filter={"file_name": "document.pdf", "page_number": 10}
            )
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

    Search with score:
        .. code-block:: python

            results = vector_store.similarity_search_with_score(query="qux", k=1)
            for doc, score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
    """

    def __init__(
        self,
        config: Optional[FireboltSettings] = None,
        embeddings: Optional[Embeddings] = None,
        use_sql_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        """Firebolt integration for LangChain Vector Store

        Args:
            config (FireboltSettings): Configuration to Firebolt Client
            embeddings (Optional[Embeddings]): Optional embeddings model. Required if use_sql_embeddings is False.
            use_sql_embeddings (bool): Whether to use SQL-based embeddings (AI_EMBED_TEXT). Defaults to True.
            kwargs (any): Other keyword arguments will pass into
                [firebolt-sdk](https://github.com/firebolt-db/firebolt-sdk-python)
                Also accepts 'embedding' (singular) for backward compatibility.
        """
        # Compatibility shim: accept both 'embedding' and 'embeddings'
        embedding_singular = kwargs.pop("embedding", None)
        if embeddings is None and embedding_singular is not None:
            # Old parameter name provided, use it and warn
            embeddings = embedding_singular
            import warnings
            warnings.warn(
                "The 'embedding' parameter is deprecated. Please use 'embeddings' (plural) instead.",
                DeprecationWarning,
                stacklevel=2
            )
        elif embeddings is not None and embedding_singular is not None:
            # Both provided, prefer 'embeddings' and warn
            import warnings
            warnings.warn(
                "Both 'embedding' and 'embeddings' parameters were provided. Using 'embeddings' (plural). "
                "The 'embedding' parameter is deprecated.",
                DeprecationWarning,
                stacklevel=2
            )
        try:
            from firebolt.client.auth import ClientCredentials
            from firebolt.db import Connection, connect
        except ImportError:
            raise ImportError(
                "Could not import firebolt-sdk python package. "
                "Please install it with `pip install firebolt-sdk`."
            )
        try:
            from tqdm import tqdm

            self.pgbar = tqdm
        except ImportError:
            # Just in case if tqdm is not installed
            self.pgbar = lambda x, **kwargs: x

        super().__init__()
        if config is not None:
            self.config = config
        else:
            self.config = FireboltSettings()
        assert self.config
        assert self.config.id and self.config.secret
        assert (
            self.config.column_map
            and self.config.database
            and self.config.table
            and self.config.metric
        )
        for k in ["id", "embedding", "document", "metadata"]:
            assert k in self.config.column_map
        # Ensure metadata is a list (can be empty)
        metadata_cols = self.config.column_map.get("metadata")
        assert isinstance(metadata_cols, list), "metadata in column_map must be a list of column names"
        # Validate that "id" is not in metadata_cols (it's the primary key column)
        if self.config.column_map['id'] in metadata_cols:
            raise ValueError(
                f"The column name {self.config.column_map['id']} cannot be in metadata_cols because it is used "
                f"as the identifier column in the Firebolt table. The identifier column is included in the metadata dictionary."
            )
        assert self.config.metric in ["vector_cosine_ops", "vector_ip_ops", "vector_l2sq_ops"]

        # Save embeddings and mode
        # Use private attribute to avoid conflict with base class property
        self._embeddings = embeddings
        self.use_sql_embeddings = use_sql_embeddings
        
        # Validate: if use_sql_embeddings is False, embeddings must be provided
        if not self.use_sql_embeddings and self._embeddings is None:
            raise ValueError(
                "embeddings must be provided when use_sql_embeddings is False"
            )
        
        # Validate: if use_sql_embeddings is True, llm_location must be provided
        if self.use_sql_embeddings and not self.config.llm_location:
            raise ValueError(
                "llm_location must be provided when use_sql_embeddings is True"
            )

        # Use AI_EMBED_TEXT for embeddings (dimension from config)
        self.dim = self.config.embedding_dimension

        # Connect to Firebolt with two connections:
        # - read_connection: for similarity_search and other read operations (autocommit=True, default)
        # - write_connection: for MERGE, DELETE, DROP operations (autocommit=False)
        auth = ClientCredentials(
            client_id=self.config.id,
            client_secret=self.config.secret
        )
        
        # Base connection parameters
        base_connection_params = {
            "engine_name": self.config.engine_name,
            "database": self.config.database,
            "account_name": self.config.account_name,
            "auth": auth,
        }
        # Add api_endpoint if specified (for custom domains/environments)
        if self.config.api_endpoint:
            base_connection_params["api_endpoint"] = self.config.api_endpoint
        
        # Set connection parameters for advanced_mode and enable_vector_search_tvf
        if "additional_parameters" not in base_connection_params:
            base_connection_params["additional_parameters"] = {}
        base_connection_params["additional_parameters"]["advanced_mode"] = "1"
        base_connection_params["additional_parameters"]["enable_vector_search_tvf"] = "1"
        
        # Read connection: autocommit=True (default) for read operations
        read_connection_params = base_connection_params.copy()
        read_connection_params["autocommit"] = True
        self.read_connection = connect(**read_connection_params)
        
        # Write connection: autocommit=False for write operations (MERGE, DELETE, DROP)
        write_connection_params = base_connection_params.copy()
        write_connection_params["autocommit"] = False
        self.write_connection = connect(**write_connection_params)
        
        # Keep self.connection as alias to write_connection for backward compatibility
        self.connection = self.write_connection
        self.client = self.write_connection
        
        # Check if table exists
        if self._table_exists():
            # Table exists: verify or create index
            if self.config.index:
                # Index name provided: check if it exists
                if not self._index_exists(self.config.index):
                    # Index doesn't exist: create it
                    logger.info(f"Index '{self.config.index}' does not exist, creating it")
                    self._create_index()
                else:
                    # Index exists: validate configuration matches settings
                    index_metadata = self._get_index_metadata(self.config.index)
                    self._validate_index_configuration(self.config.index, index_metadata)
            else:
                # No index name: auto-detect (existing behavior)
                self.config.index = self._get_index_from_db()
                # Validate auto-detected index configuration
                index_metadata = self._get_index_metadata(self.config.index)
                self._validate_index_configuration(self.config.index, index_metadata, prefix="Auto-detected index")
        else:
            # Table doesn't exist: create table and index
            self._create_table()
            # Generate index name if not provided
            if not self.config.index:
                self.config.index = f"{self.config.table}_index"
            self._create_index()

        # Set distance ordering and function based on metric
        if self.config.metric == "vector_cosine_ops":
            self.distance_function = "VECTOR_COSINE_DISTANCE"
        elif self.config.metric == "vector_ip_ops":
            self.distance_function = "1 - VECTOR_INNER_PRODUCT"
        elif self.config.metric == "vector_l2sq_ops":
            self.distance_function = "VECTOR_SQUARED_EUCLIDEAN_DISTANCE"

        # Enable get_by_ids support
        self.has_get_by_ids = True
        
        # Validate LOCATION object exists if using SQL embeddings
        if self.use_sql_embeddings and self.config.llm_location:
            if not self._location_exists(self.config.llm_location):
                raise ValueError(
                    f"LOCATION object '{self.config.llm_location}' does not exist in Firebolt. "
                    f"Please create the LOCATION object before using SQL embeddings. "
                    f"See: https://docs.firebolt.io/reference-sql/commands/data-definition/create-location-bedrock"
                )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Access the query embedding object if available."""
        return self._embeddings

    def _generate_embedding_sql(self, text: str) -> str:
        """Generate SQL query to create embedding using AI_EMBED_TEXT.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            SQL query string
        """

        # Escape single quotes in text
        escaped_text = text.replace("'", "''")
        
        sql = f"""
        SELECT AI_EMBED_TEXT(
            MODEL => '{self.config.embedding_model}',
            INPUT_TEXT => '{escaped_text}',
            DIMENSION => {self.config.embedding_dimension}, 
            LOCATION => '{self.config.llm_location}'
        ) AS embedding
        """
        return sql.strip()

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text.
        
        If client-side embeddings are configured (use_sql_embeddings=False and
        self.embeddings is not None), uses the embeddings model. Otherwise, uses
        Firebolt AI_EMBED_TEXT SQL function.
        
        Args:
            text: Text to embed
            
        Returns:
            List[float]: List of floats representing the embedding vector
            
        Raises:
            ValueError: If no embedding is returned (SQL path) or if embeddings model
                       is not configured (client-side path).
            Exception: Other errors during embedding generation are logged and re-raised.
        """
        # If client-side embeddings are configured
        if not self.use_sql_embeddings and self._embeddings is not None:
            try:
                embedding = self._embeddings.embed_query(text)
                if not embedding:
                    raise ValueError(f"No embedding returned from embeddings model for text: {text[:50]}...")
                return list(embedding)
            except Exception as e:
                logger.error(f"Error generating embedding with client-side model: {e}")
                raise
        
        # Else (default): use SQL-based embedding generation (AI_EMBED_TEXT)
        # Validate llm_location is set (required for SQL embeddings)
        if not self.config.llm_location:
            raise ValueError(
                "llm_location must be set in config to use SQL embeddings (use_sql_embeddings=True)"
            )
        
        cursor = self.read_connection.cursor()
        sql = self._generate_embedding_sql(text)
        try:
            cursor.execute(sql)
            result = cursor.fetchone()
            if result and len(result) > 0:
                # Result is an array, convert to list
                embedding = result[0]
                if isinstance(embedding, (list, tuple)):
                    return list(embedding)
                # If it's a string representation, parse it
                elif isinstance(embedding, str):
                    import ast
                    return ast.literal_eval(embedding)
                else:
                    return list(embedding)
            else:
                raise ValueError(f"No embedding returned for text: {text[:50]}...")
        except Exception as e:
            logger.error(f"Error generating embedding with SQL (AI_EMBED_TEXT): {e}")
            raise
        finally:
            cursor.close()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        *,
        ids: Optional[Iterable[str]] = None,
        batch_size: Optional[int] = None,
        embeddings: Optional[List[List[float]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vector store.

        Args:
            texts: Iterable of strings to add to the vector store.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of unique IDs. If not provided, will be auto-generated.
            batch_size: Batch size for insertion.
            embeddings: Optional precomputed embeddings. If provided, these will be used instead of generating embeddings.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        # Use config batch_size if not provided
        if batch_size is None:
            batch_size = self.config.batch_size
        
        # Convert to lists for easier handling
        texts_list = list(texts)
        if not texts_list:
            return []
        
        # Validate lengths of ids/metadatas if provided
        if ids is not None:
            ids_list = list(ids)
            if len(ids_list) != len(texts_list):
                raise ValueError(
                    f"Number of ids ({len(ids_list)}) must match number of texts ({len(texts_list)})"
                )
        else:
            ids_list = None
        
        if metadatas is not None:
            if len(metadatas) != len(texts_list):
                raise ValueError(
                    f"Number of metadatas ({len(metadatas)}) must match number of texts ({len(texts_list)})"
                )
        
        # If embeddings argument is provided
        if embeddings is not None:
            # Validate len(embeddings) == len(texts)
            if len(embeddings) != len(texts_list):
                raise ValueError(
                    f"Number of embeddings ({len(embeddings)}) must match number of texts ({len(texts_list)})"
                )
            
            # Validate each vector length == self.dim
            for i, emb in enumerate(embeddings):
                if len(emb) != self.dim:
                    raise ValueError(
                        f"Embedding at index {i} has dimension {len(emb)}, expected {self.dim}"
                    )
            
            # Use unified MERGE function with precomputed embeddings
            # Generate IDs if not provided
            if ids_list is None:
                import uuid
                ids_list = [str(uuid.uuid4()) for _ in texts_list]
            
            # Handle metadatas
            if metadatas is None:
                metadatas_list = [{}] * len(texts_list)
            else:
                metadatas_list = metadatas
            
            return self._merge_documents(
                texts=texts_list,
                ids=ids_list,
                metadatas=metadatas_list,
                embeddings=embeddings,
                batch_size=batch_size
            )
        
        # Else if use_sql_embeddings is False and self._embeddings is not None
        elif not self.use_sql_embeddings and self._embeddings is not None:
            # Compute client-side embeddings in batches (chunk_size=512)
            all_embeddings = []
            chunk_size = 512
            for i in range(0, len(texts_list), chunk_size):
                batch_texts = texts_list[i:i + chunk_size]
                batch_embeddings = self._embeddings.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)
            
            # Validate dims
            for i, emb in enumerate(all_embeddings):
                if len(emb) != self.dim:
                    raise ValueError(
                        f"Computed embedding at index {i} has dimension {len(emb)}, expected {self.dim}"
                    )
            
            # Generate IDs if not provided
            if ids_list is None:
                import uuid
                ids_list = [str(uuid.uuid4()) for _ in texts_list]
            
            # Handle metadatas
            if metadatas is None:
                metadatas_list = [{}] * len(texts_list)
            else:
                metadatas_list = metadatas
            
            # Use unified MERGE function with computed embeddings
            return self._merge_documents(
                texts=texts_list,
                ids=ids_list,
                metadatas=metadatas_list,
                embeddings=all_embeddings,
                batch_size=batch_size
            )
        
        # Else (default server-side path with SQL embeddings)
        else:
            # Validate llm_location is set (required for SQL embeddings)
            if not self.config.llm_location:
                raise ValueError(
                    "llm_location must be set in config to use SQL embeddings (use_sql_embeddings=True)"
                )
            
            # Generate IDs if not provided
            if ids_list is None:
                import uuid
                ids_list = [str(uuid.uuid4()) for _ in texts_list]
            
            # Handle metadatas
            if metadatas is None:
                metadatas_list = [{}] * len(texts_list)
            else:
                metadatas_list = metadatas
            
            # Use unified MERGE function with SQL embeddings (embeddings=None triggers SQL path)
            return self._merge_documents(
                texts=texts_list,
                ids=ids_list,
                metadatas=metadatas_list,
                embeddings=None,  # None triggers SQL embeddings path
                batch_size=batch_size
            )

    def _merge_documents(
        self,
        texts: List[str],
        ids: List[str],
        metadatas: List[Dict[Any, Any]],
        embeddings: Optional[List[List[float]]] = None,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """Unified MERGE function for adding/updating documents with either precomputed or SQL-generated embeddings.
        
        Args:
            texts: List of strings to add/update.
            ids: List of unique IDs (must match length of texts).
            metadatas: List of metadata dictionaries (must match length of texts).
            embeddings: Optional list of precomputed embeddings. If None, uses SQL embeddings (AI_EMBED_TEXT).
            batch_size: Batch size for processing.
        
        Returns:
            List of IDs that were merged.
        """
        # Use config batch_size if not provided
        if batch_size is None:
            batch_size = self.config.batch_size
        
        if not texts:
            return []
        
        if len(texts) != len(ids):
            raise ValueError(f"Number of texts ({len(texts)}) must match number of ids ({len(ids)})")
        if len(texts) != len(metadatas):
            raise ValueError(f"Number of texts ({len(texts)}) must match number of metadatas ({len(metadatas)})")
        
        # If embeddings provided, validate them
        if embeddings is not None:
            if len(embeddings) != len(texts):
                raise ValueError(f"Number of embeddings ({len(embeddings)}) must match number of texts ({len(texts)})")
            for i, emb in enumerate(embeddings):
                if len(emb) != self.dim:
                    raise ValueError(
                        f"Embedding at index {i} has dimension {len(emb)}, expected {self.dim}"
                    )
        else:
            # SQL embeddings path - validate llm_location is set
            if not self.config.llm_location:
                raise ValueError(
                    "llm_location must be set in config to use SQL embeddings"
                )
        
        # Get column mappings
        id_col = self.config.column_map['id']
        document_col = self.config.column_map['document']
        embedding_col = self.config.column_map['embedding']
        metadata_cols = self.config.column_map.get("metadata", [])
        if not isinstance(metadata_cols, list):
            metadata_cols = [metadata_cols]
        
        # Build column list for MERGE
        merge_columns = [id_col, embedding_col, document_col]
        if metadata_cols:
            merge_columns.extend(metadata_cols)
        
        # Build UPDATE SET clause (excluding id)
        update_set_clause = ", ".join([f"{col} = source.{col}" for col in merge_columns if col != id_col])
        
        cursor = self.write_connection.cursor()
        inserted_ids = []
        
        try:
            # Process in batches
            for batch_start in range(0, len(texts), batch_size):
                batch_end = min(batch_start + batch_size, len(texts))
                batch_texts = texts[batch_start:batch_end]
                batch_ids = ids[batch_start:batch_end]
                batch_metadatas = metadatas[batch_start:batch_end]
                batch_embeddings = embeddings[batch_start:batch_end] if embeddings else None
                
                if embeddings is not None:
                    # Precomputed embeddings path: build VALUES clause directly
                    values_list = []
                    for text, embedding, doc_id, metadata in zip(
                        batch_texts, batch_embeddings, batch_ids, batch_metadatas
                    ):
                        # Always quote ID since the column is TEXT
                        doc_id_str = str(doc_id)
                        escaped_id = f"'{doc_id_str.replace(chr(39), chr(39)+chr(39))}'"
                        escaped_text = text.replace(chr(39), chr(39)+chr(39))
                        
                        # Format embedding as array literal
                        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
                        
                        # Build row: id, embedding, document, and metadata columns
                        row_values = [escaped_id, embedding_str, f"'{escaped_text}'"]
                        
                        # Add metadata column values
                        for col_name in metadata_cols:
                            meta_value = metadata.get(col_name)
                            if meta_value is None:
                                row_values.append("NULL")
                            elif isinstance(meta_value, str):
                                row_values.append(f"'{meta_value.replace(chr(39), chr(39)+chr(39))}'")
                            else:
                                row_values.append(str(meta_value))
                        
                        values_list.append(f"({', '.join(row_values)})")
                    
                    # Build MERGE SQL with precomputed embeddings
                    merge_sql = f"""
                        MERGE INTO {self.config.table} AS target
                        USING (VALUES {', '.join(values_list)}) AS source ({', '.join(merge_columns)})
                        ON target.{id_col} = source.{id_col}
                        WHEN MATCHED THEN UPDATE SET {update_set_clause}
                        WHEN NOT MATCHED THEN INSERT ({', '.join(merge_columns)}) VALUES ({', '.join([f"source.{col}" for col in merge_columns])})
                    """
                else:
                    # SQL embeddings path: build CTE with AI_EMBED_TEXT
                    # Build VALUES clause for input data (id, document, metadata)
                    input_values_list = []
                    input_columns = [id_col, document_col]
                    if metadata_cols:
                        input_columns.extend(metadata_cols)
                    
                    for text, doc_id, metadata in zip(batch_texts, batch_ids, batch_metadatas):
                        # Always quote ID since the column is TEXT
                        doc_id_str = str(doc_id)
                        escaped_id = doc_id_str.replace(chr(39), chr(39)+chr(39))
                        id_value = f"'{escaped_id}'"
                        escaped_text = text.replace(chr(39), chr(39)+chr(39))
                        
                        # Build input row: id, document, and metadata columns
                        input_values = [id_value, f"'{escaped_text}'"]
                        
                        # Add metadata column values
                        for col_name in metadata_cols:
                            meta_value = metadata.get(col_name)
                            if meta_value is None:
                                input_values.append("NULL")
                            elif isinstance(meta_value, str):
                                input_values.append(f"'{meta_value.replace(chr(39), chr(39)+chr(39))}'")
                            else:
                                input_values.append(str(meta_value))
                        
                        input_values_list.append(f"({', '.join(input_values)})")
                    
                    # Build MERGE SQL with CTE using AI_EMBED_TEXT

                    merge_sql = f"""
                        MERGE INTO {self.config.table} AS target
                        USING (
                          WITH
                            content_with_embedding AS (
                              SELECT 
                                {id_col},
                                {document_col},
                                AI_EMBED_TEXT (
                                  MODEL => '{self.config.embedding_model}',
                                  INPUT_TEXT => {document_col},
                                  DIMENSION => {self.config.embedding_dimension},
                                  LOCATION => '{self.config.llm_location}'
                                ) AS {embedding_col}"""
                    
                    # Add metadata columns to CTE SELECT
                    if metadata_cols:
                        merge_sql += ",\n" + ",\n".join(metadata_cols)
                    
                    merge_sql += f"""
                              FROM (VALUES {', '.join(input_values_list)}) AS input_data({', '.join(input_columns)})
                            )
                          SELECT 
                            {id_col},
                            {embedding_col},
                            {document_col}"""
                    
                    # Add metadata columns to final SELECT
                    if metadata_cols:
                        merge_sql += ",\n" + ",\n".join(metadata_cols)
                    
                    merge_sql += f"""
                          FROM content_with_embedding
                        ) AS source ({', '.join(merge_columns)})
                        ON target.{id_col} = source.{id_col}
                        WHEN MATCHED THEN UPDATE SET {update_set_clause}
                        WHEN NOT MATCHED THEN INSERT ({', '.join(merge_columns)}) VALUES ({', '.join([f"source.{col}" for col in merge_columns])})
                    """
                
                try:
                    logger.debug(f"Executing MERGE for IDs: {batch_ids}")
                    logger.debug(f"MERGE SQL: {merge_sql[:200]}...")
                    
                    cursor.execute(merge_sql)
                    rows_affected = cursor.rowcount
                    inserted_ids.extend(batch_ids)
                                        
                    embedding_type = "precomputed" if embeddings else "SQL"
                    logger.info(f"Merged batch {batch_start//batch_size + 1} ({len(batch_ids)} rows, {rows_affected} rows affected) with {embedding_type} embeddings")
                except Exception as e:
                    logger.error(f"Error merging batch: {e}")
                    logger.error(f"SQL: {merge_sql[:500]}...")
                    try:
                        cursor.execute("ROLLBACK")
                    except Exception as rollback_error:
                        logger.warning(f"Failed to rollback after MERGE error: {rollback_error}")
                    raise
            try:
                cursor.execute("COMMIT")
            except Exception as commit_error:
                logger.warning(f"Failed to commit after MERGE: {commit_error}")
                raise

            embedding_type = "precomputed" if embeddings else "SQL"
            logger.info(f"Successfully merged {len(inserted_ids)} documents with {embedding_type} embeddings")
            
        except Exception as e:
            logger.error(f"Error in _merge_documents: {e}")
            raise
        finally:
            cursor.close()
        
        return inserted_ids

    def _insert_with_precomputed_embeddings(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        ids: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """Insert texts with precomputed embeddings into the vector store.
        
        This is a convenience wrapper around _merge_documents for backward compatibility.
        
        Args:
            texts: List of strings to add to the vector store.
            embeddings: List of precomputed embedding vectors.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of unique IDs. If not provided, will be auto-generated.
            batch_size: Batch size for insertion.
        
        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        # Use config batch_size if not provided
        if batch_size is None:
            batch_size = self.config.batch_size
        
        if not texts:
            return []
        
        if len(texts) != len(embeddings):
            raise ValueError(f"Number of texts ({len(texts)}) must match number of embeddings ({len(embeddings)})")
        
        # Generate IDs if not provided
        if ids is None:
            import uuid
            ids_list = [str(uuid.uuid4()) for _ in texts]
        else:
            ids_list = list(ids)
        
        if len(ids_list) != len(texts):
            raise ValueError(f"Number of ids ({len(ids_list)}) must match number of texts ({len(texts)})")
        
        # Handle metadatas
        if metadatas is None:
            metadatas_list = [{}] * len(texts)
        elif len(metadatas) != len(texts):
            raise ValueError(f"Number of metadatas ({len(metadatas)}) must match number of texts ({len(texts)})")
        else:
            metadatas_list = metadatas
        
        # Use unified MERGE function
        return self._merge_documents(
            texts=texts,
            ids=ids_list,
            metadatas=metadatas_list,
            embeddings=embeddings,
            batch_size=batch_size
        )

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add documents to the vector store.

        Args:
            documents: List of Document objects to add.
            ids: Optional list of IDs. If not provided, will be auto-generated.
            kwargs: Additional arguments (passed to add_texts).

        Returns:
            List of IDs.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Extract IDs from Document.id if available, otherwise from metadata
        if ids is None:
            # First, try to get IDs from Document.id attribute
            doc_ids = []
            for doc in documents:
                if doc.id is not None:
                    doc_ids.append(doc.id)
                else:
                    doc_ids.append(None)
            
            # If any document has Document.id, use those (None for ones without)
            if any(doc_ids):
                # For documents without Document.id, try metadata
                for i, doc_id in enumerate(doc_ids):
                    if doc_id is None:
                        meta_id = metadatas[i].get('id') if i < len(metadatas) else None
                        if meta_id is not None:
                            doc_ids[i] = meta_id
                # Use doc_ids if we have at least some IDs
                if any(doc_ids):
                    ids = doc_ids
            else:
                # Fall back to metadata IDs
                metadata_ids = [meta.get('id') for meta in metadatas]
                # Use metadata IDs if all are present, otherwise let add_texts generate them
                if all(metadata_ids):
                    ids = metadata_ids
        
        return self.add_texts(texts=texts, metadatas=metadatas, ids=ids, **kwargs)

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embeddings: Optional[Embeddings] = None,
        config: Optional[FireboltSettings] = None,
        ids: Optional[Iterable[str]] = None,
        batch_size: Optional[int] = None,
        use_sql_embeddings: bool = True,
        **kwargs: Any,
    ) -> "Firebolt":
        """Create Firebolt vector store from documents.
        
        Args:
            documents: List of Document objects to add.
            embeddings: Optional embeddings model. Required if use_sql_embeddings is False.
                        Also accepts 'embedding' (singular) for backward compatibility.
            config: Optional Firebolt configuration. If None, will use environment variables.
            ids: Optional list of IDs. If None, will be auto-generated.
            batch_size: Batch size when transmitting data to Firebolt. Defaults to 32.
            use_sql_embeddings: Whether to use SQL-based embeddings (AI_EMBED_TEXT). Defaults to True.
            **kwargs: Additional keyword arguments (passed to __init__).
        
        Returns:
            Firebolt: A Firebolt vector store instance with the documents added.
        """
        # Compatibility shim: accept both 'embedding' and 'embeddings'
        embedding_singular = kwargs.pop("embedding", None)
        if embeddings is None and embedding_singular is not None:
            embeddings = embedding_singular
        elif embeddings is not None and embedding_singular is not None:
            # Both provided, prefer 'embeddings'
            import warnings
            warnings.warn(
                "Both 'embedding' and 'embeddings' parameters were provided. Using 'embeddings' (plural). "
                "The 'embedding' parameter is deprecated.",
                DeprecationWarning,
                stacklevel=2
            )
        
        # Use config batch_size if not provided, otherwise use default
        if batch_size is None:
            if config is not None:
                batch_size = config.batch_size
            else:
                batch_size = DEFAULT_MERGE_BATCH_SIZE
        
        # Create instance
        instance = cls(
            config=config,
            embeddings=embeddings,
            use_sql_embeddings=use_sql_embeddings,
            **kwargs
        )
        
        # Extract texts and metadatas
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Extract IDs from metadata if available and not explicitly provided
        if ids is None:
            metadata_ids = [meta.get('id') for meta in metadatas]
            if all(metadata_ids):
                ids = metadata_ids
        
        if use_sql_embeddings:
            # Use SQL embeddings (AI_EMBED_TEXT)
            instance.add_texts(texts=texts, metadatas=metadatas, ids=ids, batch_size=batch_size)
        else:
            # Use provided embeddings model
            if instance.embeddings is None:
                raise ValueError(
                    "embeddings must be provided when use_sql_embeddings is False"
                )
            
            # Compute embeddings in batches
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = instance.embeddings.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)
            
            # Insert with precomputed embeddings
            instance._insert_with_precomputed_embeddings(
                texts=texts,
                embeddings=all_embeddings,
                metadatas=metadatas,
                ids=list(ids) if ids else None,
                batch_size=batch_size
            )
        
        return instance

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embeddings: Optional[Embeddings] = None,
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        ids: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> Firebolt:
        """Create Firebolt wrapper with existing texts
        
        Creates a Firebolt vector store instance and populates it with the provided texts.
        Supports both SQL embeddings (AI_EMBED_TEXT) and client-side embeddings.

        Args:
            texts (List[str]): List of strings to be added to the vector store.
            embeddings (Optional[Embeddings]): Embeddings model. Required if use_sql_embeddings is False.
                                               Also accepts 'embedding' (singular) for backward compatibility.
            metadatas (Optional[List[Dict[Any, Any]]]): Metadata for the texts. Defaults to None.
            ids (Optional[Iterable[str]]): IDs for the texts. If None, will be auto-generated.
            **kwargs: Additional keyword arguments:
                - config (Optional[FireboltSettings]): Firebolt configuration. If None, will use environment variables.
                - batch_size (int): Batch size when transmitting data to Firebolt. Defaults to 32.
                - precomputed_embeddings (Optional[List[List[float]]]): Precomputed embeddings. If provided, these will be used.
                    **DEPRECATED**: The 'embeddings' keyword argument in kwargs used to mean precomputed vectors and is deprecated.
                    Use 'precomputed_embeddings' instead. Example:
                    ```python
                    # New (preferred):
                    Firebolt.from_texts(texts, precomputed_embeddings=[[0.1, 0.2, ...], ...])
                    
                    # Old (deprecated):
                    Firebolt.from_texts(texts, embeddings=[[0.1, 0.2, ...], ...])  # Will issue DeprecationWarning
                    ```
                - use_sql_embeddings (bool): Whether to use SQL-based embeddings (AI_EMBED_TEXT). Defaults to True.
                - text_ids (Optional[Iterable[str]]): Backward compatibility alias for 'ids'.
                - embedding (Optional[Embeddings]): Backward compatibility alias for 'embeddings' (the model).
                    **DEPRECATED**: Use 'embeddings' (plural) parameter instead.

        Returns:
            Firebolt: A Firebolt vector store instance with the texts added.
        """
        # Compatibility shim: accept both 'embedding' and 'embeddings'
        embedding_singular = kwargs.pop("embedding", None)
        if embeddings is None and embedding_singular is not None:
            embeddings = embedding_singular
        elif embeddings is not None and embedding_singular is not None:
            # Both provided, prefer 'embeddings'
            import warnings
            warnings.warn(
                "Both 'embedding' and 'embeddings' parameters were provided. Using 'embeddings' (plural). "
                "The 'embedding' parameter is deprecated.",
                DeprecationWarning,
                stacklevel=2
            )
        
        # Extract configuration from kwargs
        config = kwargs.pop("config", None)
        batch_size = kwargs.pop("batch_size", None)
        # Use config batch_size if not provided, otherwise use default
        if batch_size is None:
            if config is not None:
                batch_size = config.batch_size
            else:
                batch_size = DEFAULT_MERGE_BATCH_SIZE
        # Accept both 'precomputed_embeddings' and 'embeddings' for backward compatibility
        # Prefer 'precomputed_embeddings' if both are provided
        precomputed_embeddings = kwargs.pop("precomputed_embeddings", None)
        embeddings_kw = kwargs.pop("embeddings", None)
        if precomputed_embeddings is None and embeddings_kw is not None:
            # Old kwarg name provided, use it and warn
            precomputed_embeddings = embeddings_kw
            import warnings
            warnings.warn(
                "The 'embeddings' keyword argument for precomputed embeddings is deprecated. "
                "Please use 'precomputed_embeddings' instead. "
                "Note: 'embeddings' (plural) is now the parameter name for the Embeddings model.",
                DeprecationWarning,
                stacklevel=2
            )
        elif precomputed_embeddings is not None and embeddings_kw is not None:
            # Both provided, prefer 'precomputed_embeddings' and warn
            import warnings
            warnings.warn(
                "Both 'precomputed_embeddings' and 'embeddings' keyword arguments were provided "
                "for precomputed vectors. Using 'precomputed_embeddings'. "
                "The 'embeddings' kwarg for precomputed vectors is deprecated.",
                DeprecationWarning,
                stacklevel=2
            )
        use_sql_embeddings = kwargs.pop("use_sql_embeddings", True)
        # Handle backward compatibility: text_ids -> ids
        text_ids = kwargs.pop("text_ids", None)
        if ids is None and text_ids is not None:
            ids = text_ids
        
        # Create Firebolt instance
        if config is None:
            config = FireboltSettings()
        
        instance = cls(
            config=config,
            embeddings=embeddings,
            use_sql_embeddings=use_sql_embeddings,
            **kwargs
        )
        
        # Add texts to the vector store
        # If precomputed embeddings provided, they will be used
        # If use_sql_embeddings=False and embeddings model provided, embeddings will be computed client-side
        # If use_sql_embeddings=True, embeddings will be generated using AI_EMBED_TEXT in the database
        instance.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            batch_size=batch_size,
            embeddings=precomputed_embeddings
        )
        
        return instance

    def _get_index_from_db(self) -> str:
        """Retrieve the vector index name from the database by querying information_schema.indexes.
        
        Returns:
            str: The name of the vector search index for the table.
            
        Raises:
            ValueError: If no vector search index is found for the table, or if multiple indexes are found.
        """
        cursor = self.read_connection.cursor()
        try:
            # Query information_schema to find vector search index for the table
            query = f"""
                SELECT index_name
                FROM information_schema.indexes
                WHERE index_type = 'vector_search' AND table_name = '{self.config.table}'
            """
            cursor.execute(query)
            results = cursor.fetchall()
            
            if not results:
                raise ValueError(
                    f"No vector search index found for table '{self.config.table}'. "
                    f"Please create a vector search index on the table or specify index explicitly."
                )
            if len(results) > 1:
                raise ValueError(
                    f"Multiple vector search indexes found for table '{self.config.table}'. "
                    f"Please specify index explicitly to disambiguate."
                )
            
            index_name = results[0][0]
            logger.info(f"Retrieved index '{index_name}' from database for table '{self.config.table}'")
            return index_name
        except Exception as e:
            logger.error(f"Error retrieving index from database: {e}")
            raise
        finally:
            cursor.close()

    def _table_exists(self) -> bool:
        """Check if the table exists in Firebolt.
        
        Returns:
            bool: True if table exists, False otherwise.
        """
        cursor = self.read_connection.cursor()
        try:
            query = f"""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_name = '{self.config.table}'
            """
            cursor.execute(query)
            results = cursor.fetchall()
            return len(results) > 0
        except Exception as e:
            logger.error(f"Error checking if table exists: {e}")
            return False
        finally:
            cursor.close()

    def _index_exists(self, index_name: str) -> bool:
        """Check if the specified index exists for the table.
        
        Args:
            index_name: Name of the index to check.
            
        Returns:
            bool: True if index exists, False otherwise.
        """
        cursor = self.read_connection.cursor()
        try:
            query = f"""
                SELECT index_name
                FROM information_schema.indexes
                WHERE index_type = 'vector_search' 
                  AND table_name = '{self.config.table}'
                  AND index_name = '{index_name}'
            """
            cursor.execute(query)
            results = cursor.fetchall()
            return len(results) > 0
        except Exception as e:
            logger.error(f"Error checking if index exists: {e}")
            return False
        finally:
            cursor.close()

    def _get_index_metadata(self, index_name: str) -> dict:
        """Get metadata for an index including metric and dimension.
        
        Args:
            index_name: Name of the index to query.
            
        Returns:
            dict: Dictionary with 'metric' and 'dimension' keys.
            
        Raises:
            ValueError: If index metadata cannot be retrieved or parsed.
        """
        cursor = self.read_connection.cursor()
        try:
            # Query for index definition - Firebolt may store this in different ways
            # Try to get the index definition from information_schema
            query = f"""
                SELECT index_definition
                FROM information_schema.indexes
                WHERE index_type = 'vector_search' 
                  AND table_name = '{self.config.table}'
                  AND index_name = '{index_name}'
            """
            cursor.execute(query)
            results = cursor.fetchall()
            
            if not results:
                raise ValueError(f"Index '{index_name}' not found in information_schema")
            
            index_definition = results[0][0] if results[0][0] else ""
            
            # Parse metric from index definition
            # Format: USING HNSW(embedding vector_cosine_ops) or similar
            metric = None
            for possible_metric in ["vector_cosine_ops", "vector_ip_ops", "vector_l2sq_ops"]:
                if possible_metric in index_definition:
                    metric = possible_metric
                    break
            
            if not metric:
                # Try alternative: query for metric directly if available
                # Some Firebolt versions might store this differently
                raise ValueError(f"Could not parse metric from index definition: {index_definition}")
            
            # Parse dimension from index definition
            # Format: WITH (dimension = 256) or similar
            dimension_match = re.search(r'dimension\s*=\s*(\d+)', index_definition, re.IGNORECASE)
            if dimension_match:
                dimension = int(dimension_match.group(1))
            else:
                # Try to get dimension from a separate query if available
                # For now, raise error if we can't parse it
                raise ValueError(f"Could not parse dimension from index definition: {index_definition}")
            
            return {
                "metric": metric,
                "dimension": dimension
            }
        except Exception as e:
            logger.error(f"Error retrieving index metadata: {e}")
            raise ValueError(f"Could not retrieve metadata for index '{index_name}': {e}")
        finally:
            cursor.close()

    def _create_table(self) -> None:
        """Create the table if it doesn't exist based on column_map configuration.
        
        Raises:
            Exception: If table creation fails.
        """
        cursor = self.write_connection.cursor()
        try:
            # Build column definitions
            columns = [
                f"{self.config.column_map['id']} TEXT",
                f"{self.config.column_map['document']} TEXT",
                f"{self.config.column_map['embedding']} ARRAY(DOUBLE PRECISION NOT NULL) NOT NULL"
            ]
            
            # Add metadata columns from column_map
            metadata_cols = self.config.column_map.get('metadata', [])
            if isinstance(metadata_cols, list):
                for col in metadata_cols:
                    # Default to TEXT type for metadata columns
                    columns.append(f"{col} TEXT")
            
            columns_sql = ",\n                ".join(columns)
            
            create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {self.config.table} (
                    {columns_sql}
                ) PRIMARY INDEX {self.config.column_map['id']}
            """
            
            cursor.execute(create_table_sql)
            cursor.execute("COMMIT")
            logger.info(f"Created table '{self.config.table}'")
        except Exception as e:
            cursor.execute("ROLLBACK")
            logger.error(f"Error creating table '{self.config.table}': {e}")
            raise
        finally:
            cursor.close()

    def _create_index(self) -> None:
        """Create the vector search index if it doesn't exist.
        
        Raises:
            Exception: If index creation fails (except for "already exists" errors).
        """
        cursor = self.write_connection.cursor()
        try:
            index_name = self.config.index
            metric = self.config.metric
            dimension = self.config.embedding_dimension
            embedding_col = self.config.column_map['embedding']
            
            create_index_sql = f"""
                CREATE INDEX {index_name}
                ON {self.config.table}
                USING HNSW({embedding_col} {metric}) WITH (dimension = {dimension})
            """
            
            cursor.execute(create_index_sql)
            cursor.execute("COMMIT")
            logger.info(f"Created vector search index '{index_name}' on table '{self.config.table}'")
        except Exception as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg or "duplicate" in error_msg:
                logger.info(f"Index '{self.config.index}' already exists")
            else:
                cursor.execute("ROLLBACK")
                logger.error(f"Error creating index '{self.config.index}': {e}")
                raise
        finally:
            cursor.close()

    def _validate_index_configuration(self, index_name: str, index_metadata: dict, prefix: str = "Index") -> None:
        """Validate that index configuration matches the settings.
        
        Args:
            index_name: Name of the index to validate.
            index_metadata: Dictionary with 'metric' and 'dimension' keys from the index.
            prefix: Prefix for error messages (e.g., "Index" or "Auto-detected index").
            
        Raises:
            ValueError: If metric or dimension doesn't match configuration.
        """
        if index_metadata['metric'] != self.config.metric:
            raise ValueError(
                f"{prefix} '{index_name}' distance metric '{index_metadata['metric']}' "
                f"does not match configured metric '{self.config.metric}'"
            )
        if index_metadata['dimension'] != self.config.embedding_dimension:
            raise ValueError(
                f"{prefix} '{index_name}' dimension {index_metadata['dimension']} "
                f"does not match configured embedding_dimension {self.config.embedding_dimension}"
            )

    def _location_exists(self, location_name: str) -> bool:
        """Check if the LOCATION object exists in Firebolt.
        
        Args:
            location_name: Name of the LOCATION object to check.
            
        Returns:
            bool: True if LOCATION exists, False otherwise.
        """
        cursor = self.read_connection.cursor()
        try:
            query = f"""
                SELECT COUNT(*) 
                FROM information_schema.locations 
                WHERE location_name = '{location_name}'
            """
            cursor.execute(query)
            result = cursor.fetchone()
            return result[0] > 0 if result else False
        except Exception as e:
            logger.error(f"Error checking if LOCATION exists: {e}")
            return False
        finally:
            cursor.close()

    def __repr__(self) -> str:
        """Text representation for Firebolt Vector Store, prints connection info
            and schemas. Easy to use with `str(Firebolt())`

        Returns:
            repr: string to show connection info and data schema
        """
        _repr = f"\033[92m\033[1m{self.config.database}.{self.config.table}\033[0m\n\n"
        _repr += f"\033[1mEngine: {self.config.engine_name}\033[0m\n"
        _repr += f"\033[1mAccount: {self.config.account_name}\033[0m\n"
        _repr += f"\033[1mTable: {self.config.table}\033[0m\n"
        _repr += f"\033[1mIndex: {self.config.index}\033[0m\n"
        return _repr

    def _build_filter_clause(self, filter: Dict[str, Any]) -> str:
        """Convert a dictionary filter to a SQL WHERE clause.
        
        Validates filter keys against configured metadata columns, escapes string values,
        and supports equality, IS NULL, and IN clauses. Multiple conditions are combined
        with AND.
        
        Args:
            filter: Dictionary with metadata column names as keys and values to match.
                   Values can be:
                   - Simple values for equality: {"file_name": "document.pdf"}
                   - None for IS NULL: {"file_name": None}
                   - Lists for IN clauses: {"file_name": ["doc1.pdf", "doc2.pdf"]}
                   - Multiple conditions are combined with AND
                   Example: {"file_name": "document.pdf", "page_number": 10}
        
        Returns:
            str: SQL WHERE clause string (without the leading WHERE keyword).
                 Example: "file_name = 'document.pdf' AND page_number = 10"
        
        Raises:
            ValueError: If any filter key is not a valid metadata column.
        """
        conditions = []
        
        # Get metadata columns from column_map to validate filter keys
        metadata_cols = self.config.column_map.get("metadata", [])
        if not isinstance(metadata_cols, list):
            metadata_cols = [metadata_cols]
        
        for key, value in filter.items():
            # Validate that the key is a valid metadata column
            if key not in metadata_cols:
                raise ValueError(
                    f"Filter key '{key}' is not a valid metadata column. "
                    f"Valid columns: {metadata_cols}"
                )
            
            # Column name is validated and safe (comes from column_map)
            # No need to escape column names as they're validated against metadata_cols
            
            if isinstance(value, list):
                # Handle IN clause for lists
                if not value:
                    # Empty list means no matches
                    conditions.append("1 = 0")  # Always false
                else:
                    # Escape and quote each value
                    escaped_values = []
                    for v in value:
                        if isinstance(v, str):
                            escaped_v = v.replace("'", "''")
                            escaped_values.append(f"'{escaped_v}'")
                        else:
                            escaped_values.append(str(v))
                    conditions.append(f"{key} IN ({', '.join(escaped_values)})")
            else:
                # Handle equality for single values
                if isinstance(value, str):
                    escaped_value = value.replace("'", "''")
                    conditions.append(f"{key} = '{escaped_value}'")
                elif value is None:
                    conditions.append(f"{key} IS NULL")
                else:
                    conditions.append(f"{key} = {value}")
        
        return " AND ".join(conditions)
    
    def _build_query_sql(
        self, q_emb: List[float], topk: int, filter: Optional[Dict[str, Any]] = None
    ) -> str:
        """Construct an SQL query for performing a similarity search.

        Builds a SELECT query with id/document/metadata columns plus distance alias 'dist',
        uses vector_search table-valued function, optionally injects WHERE clause from filter,
        and orders results by distance according to the configured metric.

        Args:
            q_emb: The query vector as a list of floats. Will be rendered as comma-separated
                   literal format: [val1,val2,val3,...]
            topk: The number of top similar items to retrieve.
            filter: Optional dictionary for filtering by metadata.
                   Keys should be metadata column names, values can be:
                   - Simple values for equality: {"file_name": "document.pdf"}
                   - Lists for IN clauses: {"file_name": ["doc1.pdf", "doc2.pdf"]}
                   - None for IS NULL: {"file_name": None}
                   - Multiple conditions are combined with AND

        Returns:
            str: SQL query string that:
                 - SELECTs id, document, metadata columns, and distance calculated based on the metric
                 - FROM vector_search(INDEX index, [q_emb], topk, 16)
                 - WHERE clause (if filter provided, built via _build_filter_clause)
                 - ORDER BY dist
        """
        q_emb_str = ",".join(map(str, q_emb))

        # Get id column from column_map (required)
        id_col = self.config.column_map['id']
        
        # Get metadata columns from column_map
        metadata_cols = self.config.column_map.get("metadata", [])
        if not isinstance(metadata_cols, list):
            metadata_cols = [metadata_cols]  # Backward compatibility: convert single string to list
        
        # Build SELECT clause: id, document, metadata columns (if any), dist
        select_columns = [
            id_col,
            self.config.column_map['document']
        ]
        if metadata_cols:
            select_columns.extend(metadata_cols)
        
        distance_func = f"{self.distance_function}({self.config.column_map['embedding']}, [{q_emb_str}])"
        select_columns.append(f"{distance_func} AS dist")
        
        # Build WHERE clause if filter is provided
        where_clause = ""
        if filter:
            filter_conditions = self._build_filter_clause(filter)
            where_clause = f"WHERE {filter_conditions}"
        
        q_str = f"""
            SELECT 
                {', '.join(select_columns)}
            FROM vector_search( INDEX {self.config.index}, [{q_emb_str}], {topk}, 16 )
            {where_clause}
            ORDER BY dist 
        """
        return q_str

    def _build_search_query_with_text_sql(
        self, query_text: str, topk: int, filter: Optional[Dict[str, Any]] = None
    ) -> str:
        """Construct a single SQL query that embeds the query text and performs vector search.

        Uses a CTE (Common Table Expression) to combine AI_EMBED_TEXT and vector_search
        into a single database request, avoiding the round-trip of fetching the embedding first.

        Args:
            query_text: The text query to embed and search for.
            topk: The number of top similar items to retrieve.
            filter: Optional dictionary for filtering by metadata.
                   Keys should be metadata column names, values can be:
                   - Simple values for equality: {"file_name": "document.pdf"}
                   - Lists for IN clauses: {"file_name": ["doc1.pdf", "doc2.pdf"]}
                   - None for IS NULL: {"file_name": None}
                   - Multiple conditions are combined with AND

        Returns:
            str: SQL query string with CTE that:
                 - Computes query embedding using AI_EMBED_TEXT in a CTE
                 - SELECTs id, document, metadata columns, and distance
                 - FROM vector_search using the CTE embedding
                 - WHERE clause (if filter provided)
                 - ORDER BY dist
        """
        # Escape single quotes in query text
        escaped_text = query_text.replace("'", "''")

        # Get id column from column_map (required)
        id_col = self.config.column_map['id']
        
        # Get metadata columns from column_map
        metadata_cols = self.config.column_map.get("metadata", [])
        if not isinstance(metadata_cols, list):
            metadata_cols = [metadata_cols]  # Backward compatibility: convert single string to list
        
        # Build SELECT clause: id, document, metadata columns (if any), dist
        select_columns = [
            id_col,
            self.config.column_map['document']
        ]
        if metadata_cols:
            select_columns.extend(metadata_cols)
        
        # Distance function using the CTE embedding
        distance_func = f"{self.distance_function}({self.config.column_map['embedding']}, (SELECT emb FROM query_embedding))"
        select_columns.append(f"{distance_func} AS dist")
        
        # Build WHERE clause if filter is provided
        where_clause = ""
        if filter:
            filter_conditions = self._build_filter_clause(filter)
            where_clause = f"WHERE {filter_conditions}"
        
        q_str = f"""
            WITH query_embedding AS (
                SELECT AI_EMBED_TEXT(
                    MODEL => '{self.config.embedding_model}',
                    INPUT_TEXT => '{escaped_text}',
                    DIMENSION => {self.config.embedding_dimension},
                    LOCATION => '{self.config.llm_location}'
                ) AS emb
            )
            SELECT 
                {', '.join(select_columns)}
            FROM vector_search( INDEX {self.config.index}, (SELECT emb FROM query_embedding), {topk}, 16 )
            {where_clause}
            ORDER BY dist 
        """
        return q_str

    def _execute_text_search(
        self, query_text: str, k: int, filter: Optional[Dict[str, Any]] = None, with_score: bool = False
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
        """Execute a text-based similarity search using a single CTE query.

        Combines embedding generation and vector search into one database request.

        Args:
            query_text: The text query to search for.
            k: Number of results to return.
            filter: Optional metadata filter.
            with_score: If True, returns (Document, score) tuples. If False, returns Documents only.

        Returns:
            List of Documents or List of (Document, score) tuples depending on with_score.
        """
        q_str = self._build_search_query_with_text_sql(query_text, k, filter=filter)
        cursor = self.read_connection.cursor()
        try:
            cursor.execute(q_str)
            results = []
            
            # Get metadata columns from column_map
            metadata_cols = self.config.column_map.get("metadata", [])
            if not isinstance(metadata_cols, list):
                metadata_cols = [metadata_cols]  # Backward compatibility
            
            for row in cursor.fetchall():
                # Row structure: [id, document, metadata_cols..., dist]
                row_idx = 0
                
                # Extract id (required)
                doc_id = row[row_idx] if len(row) > row_idx else None
                row_idx += 1
                
                # Extract document content
                doc_content = row[row_idx] if len(row) > row_idx else ""
                row_idx += 1
                
                # Build metadata dictionary
                metadata = {}
                
                # Extract metadata columns
                if metadata_cols:
                    for col_name in metadata_cols:
                        if len(row) > row_idx:
                            value = row[row_idx]
                            if value is not None:
                                metadata[col_name] = value
                            row_idx += 1
                else:
                    # If no metadata columns specified, extract all remaining columns
                    # (except the last one which is distance)
                    col_names = []
                    if hasattr(cursor, 'description') and cursor.description:
                        col_names = [desc[0] for desc in cursor.description[2:-1]] if len(cursor.description) > 3 else []
                    
                    while row_idx < len(row) - 1:  # -1 to skip distance column
                        value = row[row_idx]
                        col_name = col_names[row_idx - 2] if row_idx - 2 < len(col_names) else f"_col_{row_idx - 2}"
                        if value is not None:
                            metadata[col_name] = value
                        row_idx += 1
                
                # Always set metadata[id_col] from doc_id
                if doc_id is not None:
                    metadata[self.config.column_map['id']] = self._try_convert_id_to_int(doc_id)
                
                doc = Document(
                    page_content=doc_content,
                    metadata=metadata
                )
                if doc_id is not None:
                    doc.id = str(doc_id)
                
                if with_score:
                    # The distance/score is the last column
                    score = float(row[row_idx]) if len(row) > row_idx else 0.0
                    results.append((doc, score))
                else:
                    results.append(doc)
            
            return results
        except Exception as e:
            logger.error(f"\033[91m\033[1m{type(e)}\033[0m \033[95m{str(e)}\033[0m")
            return []
        finally:
            cursor.close()

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a similarity search with Firebolt

        Args:
            query (str): Query string.
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.
            filter (Optional[Dict[str, Any]], optional): Dictionary for filtering by metadata.
                                                         Keys should be metadata column names.
                                                         Values can be:
                                                         - Simple values for equality: {"file_name": "document.pdf"}
                                                         - Lists for IN clauses: {"file_name": ["doc1.pdf", "doc2.pdf"]}
                                                         - Multiple conditions are combined with AND
                                                         Example: {"file_name": "document.pdf", "page_number": 10}
                                                         Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Document]: List of Documents with content and metadata (including 'id')
        """
        # If using SQL embeddings, use single-request CTE query for better performance
        if self.use_sql_embeddings:
            return self._execute_text_search(query, k, filter=filter, with_score=False)
        
        # Client-side embeddings: compute embedding first, then search by vector
        query_embedding = self._embeddings.embed_query(query)
        return self.similarity_search_by_vector(query_embedding, k, filter=filter, **kwargs)

    def _try_convert_id_to_int(self, id_value: Any) -> Any:
        """Try to convert an ID value back to integer if it looks like one.
        
        This helps preserve integer IDs when reading from TEXT columns.
        Only converts if the value is a string that represents a valid integer.
        
        Args:
            id_value: The ID value (could be string, int, etc.)
            
        Returns:
            The original value if it's already an int, or converted to int if it's a string
            representing an integer, otherwise returns the original value.
        """
        if isinstance(id_value, int):
            return id_value
        if isinstance(id_value, str):
            # Try to convert to int if it looks like an integer
            # Check if it's a valid integer (no decimal point, no leading zeros unless it's "0")
            try:
                # Only convert if it's a simple integer representation
                if id_value.isdigit() or (id_value.startswith('-') and id_value[1:].isdigit()):
                    return int(id_value)
            except (ValueError, AttributeError):
                pass
        return id_value

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a similarity search with Firebolt by vectors

        Uses _build_query_sql to construct vector_search(...) SQL, executes the query,
        and maps rows to Document objects with content and metadata (including id and
        configured metadata columns). Results are ordered by distance based on the configured metric.

        Args:
            embedding (List[float]): query vector
            k (int, optional): Top K neighbors to retrieve. Defaults to 20.
            filter (Optional[Dict[str, Any]], optional): Dictionary for filtering by metadata.
                                                         Keys should be metadata column names.
                                                         Values can be:
                                                         - Simple values for equality: {"file_name": "document.pdf"}
                                                         - Lists for IN clauses: {"file_name": ["doc1.pdf", "doc2.pdf"]}
                                                         - Multiple conditions are combined with AND
                                                         Example: {"file_name": "document.pdf", "page_number": 10}
                                                         Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Document]: List of Documents with content and metadata (including 'id' and configured metadata columns),
                           ordered by distance.
        """
        # Use _build_query_sql to construct vector_search(...) SQL
        q_str = self._build_query_sql(embedding, k, filter=filter)
        cursor = self.read_connection.cursor()
        try:
            # Now execute the SELECT query (enable_vector_search_tvf is set at connection time)
            cursor.execute(q_str)
            results = []
            
            # Get id column from column_map (required)
            id_col = self.config.column_map['id']
            
            # Get metadata columns from column_map
            metadata_cols = self.config.column_map.get("metadata", [])
            if not isinstance(metadata_cols, list):
                metadata_cols = [metadata_cols]  # Backward compatibility
            
            for row in cursor.fetchall():
                # Row structure: [id, document, metadata_cols..., dist]
                row_idx = 0
                
                # Extract id (required)
                doc_id = row[row_idx] if len(row) > row_idx else None
                row_idx += 1
                
                # Extract document content
                doc_content = row[row_idx] if len(row) > row_idx else ""
                row_idx += 1
                
                # Build metadata dictionary with id and all metadata columns (if any)
                metadata = {}
                
                # Extract metadata columns
                if metadata_cols:
                    for col_name in metadata_cols:
                        if len(row) > row_idx:
                            value = row[row_idx]
                            if value is not None:
                                # Use the column name as the metadata key
                                metadata[col_name] = value
                            row_idx += 1
                else:
                    # If no metadata columns specified, extract all remaining columns
                    # (except the last one which is distance)
                    # This is useful for testing/mocking scenarios
                    # Try to get column names from cursor description if available
                    col_names = []
                    if hasattr(cursor, 'description') and cursor.description:
                        # Skip id and document columns (first 2), get rest except distance (last)
                        col_names = [desc[0] for desc in cursor.description[2:-1]] if len(cursor.description) > 3 else []
                    
                    # Extract all columns between document and distance
                    while row_idx < len(row) - 1:  # -1 to skip distance column
                        value = row[row_idx]
                        col_name = col_names[row_idx - 2] if row_idx - 2 < len(col_names) else f"_col_{row_idx - 2}"
                        if value is not None:
                            metadata[col_name] = value
                        row_idx += 1
                
                # Always set metadata[id_col] from doc_id (the primary key column)
                # This preserves the document ID in metadata for consistency
                if doc_id is not None:
                    # Try to preserve integer type if the ID looks like an integer
                    # This helps when integer IDs are stored in TEXT columns
                    metadata[self.config.column_map['id']] = self._try_convert_id_to_int(doc_id)
                
                doc = Document(
                    page_content=doc_content,
                    metadata=metadata
                )
                # Set id attribute - use the doc_id (string) for Document.id
                # but keep the original type in metadata['id']
                if doc_id is not None:
                    doc.id = str(doc_id)  # Document.id should be string
                results.append(doc)
            return results
        except Exception as e:
            logger.error(f"\033[91m\033[1m{type(e)}\033[0m \033[95m{str(e)}\033[0m")
            return []
        finally:
            cursor.close()

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a similarity search with Firebolt by vectors, returning documents with scores

        Uses _build_query_sql to construct vector_search(...) SQL, executes the query,
        and maps rows to tuples of (Document, score) where score is the distance returned
        by the database. Results are ordered by distance (handled by SQL ORDER BY dist {self.dist_order}).

        Args:
            embedding (List[float]): Query vector.
            k (int, optional): Top K neighbors to retrieve. Defaults to 20.
            filter (Optional[Dict[str, Any]], optional): Dictionary for filtering by metadata.
                                                         Keys should be metadata column names.
                                                         Values can be:
                                                         - Simple values for equality: {"file_name": "document.pdf"}
                                                         - Lists for IN clauses: {"file_name": ["doc1.pdf", "doc2.pdf"]}
                                                         - Multiple conditions are combined with AND
                                                         Example: {"file_name": "document.pdf", "page_number": 10}
                                                         Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Tuple[Document, float]]: List of (Document, score) tuples where:
            - Document contains page_content and metadata (including 'id' and configured metadata columns)
            - score is the distance returned by the database (float)
            
        Note:
            Score interpretation depends on the metric:
            - For cosine/l2sq: Lower scores indicate higher similarity (results ordered ASC)
            - For inner product (ip): we use (1 - VECTOR_INNER_PRODUCT), so lower scores continue to indicate higher similarity (results ordered DESC)
        """
        # Build and execute SQL query
        q_str = self._build_query_sql(embedding, k, filter=filter)
        cursor = self.read_connection.cursor()
        try:
            # Now execute the SELECT query (enable_vector_search_tvf is set at connection time)
            cursor.execute(q_str)
            results = []
            
            # Get id column from column_map (required)
            id_col = self.config.column_map['id']
            
            # Get metadata columns from column_map
            metadata_cols = self.config.column_map.get("metadata", [])
            if not isinstance(metadata_cols, list):
                metadata_cols = [metadata_cols]  # Backward compatibility
            
            for row in cursor.fetchall():
                # Row structure: [id, document, metadata_cols..., dist]
                row_idx = 0
                
                # Extract id (required)
                doc_id = row[row_idx] if len(row) > row_idx else None
                row_idx += 1
                
                # Extract document content
                doc_content = row[row_idx] if len(row) > row_idx else ""
                row_idx += 1
                
                # Build metadata dictionary with id and all metadata columns (if any)
                metadata = {}
                if doc_id is not None:
                    metadata[self.config.column_map['id']] = doc_id
                
                # Extract metadata columns
                if metadata_cols:
                    for col_name in metadata_cols:
                        if len(row) > row_idx:
                            value = row[row_idx]
                            if value is not None:
                                # Use the column name as the metadata key
                                metadata[col_name] = value
                            row_idx += 1
                else:
                    # If no metadata columns specified, extract all remaining columns
                    # (except the last one which is distance)
                    # This is useful for testing/mocking scenarios
                    # Try to get column names from cursor description if available
                    col_names = []
                    if hasattr(cursor, 'description') and cursor.description:
                        # Skip id and document columns (first 2), get rest except distance (last)
                        col_names = [desc[0] for desc in cursor.description[2:-1]] if len(cursor.description) > 3 else []
                    
                    # Extract all columns between document and distance
                    while row_idx < len(row) - 1:  # -1 to skip distance column
                        value = row[row_idx]
                        col_name = col_names[row_idx - 2] if row_idx - 2 < len(col_names) else f"_col_{row_idx - 2}"
                        if value is not None:
                            metadata[col_name] = value
                        row_idx += 1
                
                # The distance/score is the last column
                score = float(row[row_idx]) if len(row) > row_idx else 0.0
                
                doc = Document(
                    page_content=doc_content,
                    metadata=metadata
                )
                # Set id attribute - use the doc_id (string) for Document.id
                # but keep the original type in metadata['id']
                if doc_id is not None:
                    doc.id = str(doc_id)  # Document.id should be string
                results.append((doc, score))
            return results
        except Exception as e:
            logger.error(f"\033[91m\033[1m{type(e)}\033[0m \033[95m{str(e)}\033[0m")
            return []
        finally:
            cursor.close()

    def similarity_search_with_score(
        self,
        query: Optional[str] = None,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a similarity search with Firebolt, returning documents with scores

        Same dual input behavior as similarity_search: accepts either query (string) or
        explicit embedding (list of floats). Computes or retrieves embedding accordingly,
        runs SQL, and parses results into list of tuples (Document, score) where score
        is the distance returned by the database.

        Args:
            query (Optional[str]): Query string. Either query or embedding must be provided.
            k (int, optional): Top K neighbors to retrieve. Defaults to 20.
            filter (Optional[Dict[str, Any]], optional): Dictionary for filtering by metadata.
                                                         Keys should be metadata column names.
                                                         Values can be:
                                                         - Simple values for equality: {"file_name": "document.pdf"}
                                                         - Lists for IN clauses: {"file_name": ["doc1.pdf", "doc2.pdf"]}
                                                         - Multiple conditions are combined with AND
                                                         Example: {"file_name": "document.pdf", "page_number": 10}
                                                         Defaults to None.
            embedding (Optional[List[float]], optional): Precomputed query embedding. Either query or embedding must be provided.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Tuple[Document, float]]: List of (Document, score) tuples where:
            - Document contains page_content and metadata (including 'id' and configured metadata columns)
            - score is the distance returned by the database (float)
            
        Note:
            Score interpretation depends on the metric:
            - For cosine/l2sq: Lower scores indicate higher similarity (results ordered ASC)
            - For inner product (ip): We use (1 - VECTOR_INNER_PRODUCT), so lower scores continue to indicate higher similarity (results ordered ASC)
        """
        # Require at least one of query or embedding
        if query is None and embedding is None:
            raise ValueError("Either 'query' or 'embedding' must be provided")
        
        # If embedding is provided, use it directly
        if embedding is not None:
            return self.similarity_search_with_score_by_vector(embedding, k, filter=filter, **kwargs)
        
        # Query is provided - use single-request CTE if using SQL embeddings
        if self.use_sql_embeddings:
            return self._execute_text_search(query, k, filter=filter, with_score=True)
        
        # Client-side embeddings: compute embedding first, then search by vector
        query_embedding = self._embeddings.embed_query(query)
        return self.similarity_search_with_score_by_vector(query_embedding, k, filter=filter, **kwargs)

    def get_by_ids(self, ids: List[str]) -> List[Document]:
        """Get documents by their IDs.
        
        Retrieves documents from the vector store by their IDs. Returns documents
        in the same order as the input IDs. If an ID is not found, it will be skipped
        (the returned list may be shorter than the input list).
        
        Args:
            ids: List of document IDs to retrieve.
            
        Returns:
            List[Document]: List of Document objects in the same order as the input IDs.
                          Documents that were not found are omitted from the result.
        """
        if not ids:
            return []
        
        cursor = self.read_connection.cursor()
        try:
            # Get column mappings
            id_col = self.config.column_map['id']
            document_col = self.config.column_map['document']
            metadata_cols = self.config.column_map.get("metadata", [])
            if not isinstance(metadata_cols, list):
                metadata_cols = [metadata_cols]
            
            # Escape IDs for SQL IN clause
            # Always quote IDs since the column is TEXT (even if id_val is an integer)
            escaped_ids = []
            for id_val in ids:
                id_val_str = str(id_val)
                escaped_id = id_val_str.replace(chr(39), chr(39)+chr(39))
                escaped_ids.append(f"'{escaped_id}'")
            ids_str = ", ".join(escaped_ids)
            
            # Build SELECT query to get documents by IDs
            select_columns = [id_col, document_col]
            if metadata_cols:
                select_columns.extend(metadata_cols)
            
            select_sql = f"""
                SELECT {', '.join(select_columns)}
                FROM {self.config.table}
                WHERE {id_col} IN ({ids_str})
            """
            
            cursor.execute(select_sql)
            results = cursor.fetchall()
            
            # Create a dictionary mapping ID to Document for quick lookup
            id_to_doc = {}
            for row in results:
                row_idx = 0
                
                # Extract id
                doc_id = row[row_idx] if len(row) > row_idx else None
                row_idx += 1
                
                # Extract document content
                doc_content = row[row_idx] if len(row) > row_idx else ""
                row_idx += 1
                
                # Build metadata dictionary
                metadata = {}
                
                # Extract metadata columns first (these take precedence)
                for col_name in metadata_cols:
                    if len(row) > row_idx:
                        value = row[row_idx]
                        if value is not None:
                            # Use the column name as the metadata key
                            metadata[col_name] = value
                        row_idx += 1
                
                # Always set metadata[id_col] from doc_id (the primary key column)
                # This preserves the document ID in metadata for consistency
                if doc_id is not None:
                    # Try to preserve integer type if the ID looks like an integer
                    metadata[self.config.column_map['id']] = self._try_convert_id_to_int(doc_id)
                
                # Create Document
                doc = Document(
                    page_content=doc_content,
                    metadata=metadata
                )
                if doc_id is not None:
                    doc.id = str(doc_id)  # Document.id should be string
                
                # Use string representation of ID as key for lookup
                id_to_doc[str(doc_id)] = doc
            
            # Return documents in the same order as input IDs
            # If an ID is not found, it will be omitted
            documents = []
            for id_val in ids:
                id_str = str(id_val)
                if id_str in id_to_doc:
                    documents.append(id_to_doc[id_str])
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in get_by_ids: {e}")
            raise
        finally:
            cursor.close()

    async def aget_by_ids(self, ids: List[str]) -> List[Document]:
        """Asynchronously get documents by their IDs.
        
        Retrieves documents from the vector store by their IDs. Returns documents
        in the same order as the input IDs. If an ID is not found, it will be skipped
        (the returned list may be shorter than the input list).
        
        Args:
            ids: List of document IDs to retrieve.
            
        Returns:
            List[Document]: List of Document objects in the same order as the input IDs.
                          Documents that were not found are omitted from the result.
        """
        return await asyncio.to_thread(self.get_by_ids, ids)

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        delete_all: bool = False,
        **kwargs: Any,
    ) -> Optional[bool]:
        """
        Delete records from the vector store.

        Args:
            ids: Optional list of IDs to delete. If provided, deletes records matching these IDs.
            filter: Optional dictionary for filtering by metadata. Keys should be metadata column names.
                   Values can be:
                   - Simple values for equality: {"file_name": "document.pdf"}
                   - Lists for IN clauses: {"file_name": ["doc1.pdf", "doc2.pdf"]}
                   - Multiple conditions are combined with AND
            delete_all: If True, deletes all records. Requires explicit flag for safety.
            **kwargs: Additional keyword arguments.

        Returns:
            Optional[bool]: True if deletion was successful, False if no-op (e.g., empty ids list).

        Raises:
            ValueError: If ids, filter, and delete_all are all None/False (explicit deletion required).
        """
        # Validate that at least one deletion method is specified
        if ids is None and filter is None and not delete_all:
            raise ValueError(
                "Explicit deletion required. Provide one of: ids, filter, or delete_all=True"
            )
        
        # If ids is an empty list, return False (no-op)
        if ids is not None and len(ids) == 0:
            logger.warning("delete called with empty ids list, no action taken")
            return False
        
        cursor = self.write_connection.cursor()
        try:
            # Get id column name from column_map
            id_col = self.config.column_map['id']
            
            # Build DELETE statement
            if delete_all:
                # Delete all records (with warning)
                logger.warning(f"Deleting all records from {self.config.table} (delete_all=True)")
                delete_sql = f"DELETE FROM {self.config.table}"
            elif filter is not None:
                # Delete using filter
                where_clause = self._build_filter_clause(filter)
                delete_sql = f"""
                    DELETE FROM {self.config.table}
                    WHERE {where_clause}
                """
            elif ids is not None:
                # Delete using IDs
                # Always quote IDs since the column is TEXT (even if id_val is an integer)
                escaped_ids = []
                for id_val in ids:
                    id_val_str = str(id_val)
                    escaped_id = id_val_str.replace(chr(39), chr(39)+chr(39))
                    escaped_ids.append(f"'{escaped_id}'")
                ids_str = ", ".join(escaped_ids)
                
                delete_sql = f"""
                    DELETE FROM {self.config.table}
                    WHERE {id_col} IN ({ids_str})
                """                
            # Execute delete statement
            cursor.execute(delete_sql)
            deleted_count = cursor.rowcount
            cursor.execute("COMMIT")
            logger.info(f"Deleted all {deleted_count} records from {self.config.table}")
            return True
            
        except Exception as e:
            logger.error(f"Error in delete: {e}")
            try:
                cursor.execute("ROLLBACK")
            except Exception as rollback_error:
                logger.warning(f"Failed to rollback after delete error: {rollback_error}")
            raise
        finally:
            cursor.close()

    def drop(self, drop_table: bool = False) -> None:
        """
        Drop the vector index and/or table.
        
        WARNING: This is a destructive operation that will permanently delete the vector
        index and table, including all stored vectors and metadata. This operation cannot
        be undone.
        
        By default, this method raises NotImplementedError to prevent accidental data loss.
        To actually drop the table and index, explicitly pass drop_table=True.
        
        Args:
            drop_table (bool): If True, drops both the vector index and table.
                              Defaults to False to prevent accidental deletion.
        
        Raises:
            NotImplementedError: If drop_table is False (default behavior to prevent accidents).
        """
        if not drop_table:
            raise NotImplementedError(
                "drop() is not implemented by default to prevent accidental data loss. "
                "To drop the vector index and table, explicitly pass drop_table=True. "
                "WARNING: This will permanently delete all data in the table and index."
            )
        
        # Destructive operation: drop index and table
        cursor = self.write_connection.cursor()
        try:
            # Drop index first (if it exists)
            # Drop table (if it exists)
            drop_table_sql = f"DROP TABLE IF EXISTS {self.config.table} CASCADE"
            logger.warning(f"Dropping table: {self.config.table}")
            cursor.execute(drop_table_sql)
            logger.info(f"Successfully dropped table: {self.config.table}")
            
            cursor.execute("COMMIT")
            
            logger.warning(
                f"Successfully dropped vector index and table. "
                f"All data has been permanently deleted."
            )
            
        except Exception as e:
            logger.error(f"Error in drop: {e}")
            try:
                cursor.execute("ROLLBACK")
            except Exception as rollback_error:
                logger.warning(f"Failed to rollback after drop error: {rollback_error}")
            raise
        finally:
            cursor.close()

    def as_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Return a retriever object that wraps this vector store.
        
        Args:
            search_type (str): Type of search to perform. Currently only "similarity" is supported.
                              Raises ValueError for any other search_type.
            search_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments to pass to search methods.
                                                      Defaults to {"k": 4}. Supports filter and other search parameters.
        
        Returns:
            FireboltRetriever: A retriever object that implements:
                - get_relevant_documents(query: str) -> List[Document]
                - get_relevant_documents_with_score(query: str) -> List[Tuple[Document, float]]
                - aget_relevant_documents(query: str) -> List[Document] (async)
                - aget_relevant_documents_with_score(query: str) -> List[Tuple[Document, float]] (async)
        
        Raises:
            ValueError: If search_type is not "similarity". Only similarity search is currently supported.
        """
        if search_type != "similarity":
            raise ValueError(f"search_type '{search_type}' not supported. Only 'similarity' is supported.")
        
        # Default search_kwargs
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        
        return FireboltRetriever(vector_store=self, search_kwargs=search_kwargs)

    def close(self) -> None:
        """Close the Firebolt connections."""
        # Close read connection
        if hasattr(self, 'read_connection') and self.read_connection:
            try:
                self.read_connection.close()
                logger.debug("Firebolt read connection closed successfully")
            except Exception as e:
                logger.warning(f"Error closing Firebolt read connection: {e}")
            finally:
                self.read_connection = None
        
        # Close write connection
        if hasattr(self, 'write_connection') and self.write_connection:
            try:
                self.write_connection.close()
                logger.debug("Firebolt write connection closed successfully")
            except Exception as e:
                logger.warning(f"Error closing Firebolt write connection: {e}")
            finally:
                self.write_connection = None
                self.connection = None
                self.client = None

    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        self.close()

    def __enter__(self) -> "Firebolt":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - close connection."""
        self.close()

    @property
    def metadata_column(self) -> List[str]:
        """Return the metadata column names as a list."""
        metadata_cols = self.config.column_map.get("metadata", [])
        if not isinstance(metadata_cols, list):
            metadata_cols = [metadata_cols]  # Backward compatibility
        return metadata_cols

