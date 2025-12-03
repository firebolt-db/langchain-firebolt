"""Integration tests for Firebolt retriever."""

import os
import pytest
from typing import Any, Dict

from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_tests.integration_tests.retrievers import RetrieversIntegrationTests

from langchain_firebolt import (
    Firebolt,
    FireboltRetriever,
    FireboltSettings,
)


class TestFireboltRetrieverIntegration(RetrieversIntegrationTests):
    """Integration tests for Firebolt retriever."""

    @pytest.fixture(autouse=True)
    def _store_pytest_request(self, request):
        """Auto-use fixture to store pytest request for retriever_constructor property."""
        self._pytest_request = request

    @pytest.fixture
    def retriever_cls(self) -> type:
        """Return the FireboltRetriever class."""
        return FireboltRetriever

    @pytest.fixture
    def vectorstore(
        self, 
        embedding_openai: FakeEmbeddings,
        firebolt_table_setup: dict
    ) -> Firebolt:
        """Create a Firebolt vector store instance for testing.
        
        Args:
            embedding_openai: The embeddings model to use.
            firebolt_table_setup: Fixture that ensures table and index exist.
        """
        # Get configuration from environment variables
        client_id = os.getenv("FIREBOLT_CLIENT_ID")
        client_secret = os.getenv("FIREBOLT_CLIENT_SECRET")
        engine_name = os.getenv("FIREBOLT_ENGINE")
        database = os.getenv("FIREBOLT_DB")
        account_name = os.getenv("FIREBOLT_ACCOUNT")
        table = firebolt_table_setup["table_name"]
        index = firebolt_table_setup["index_name"]
        metric = firebolt_table_setup.get("metric", "vector_cosine_ops")
        llm_location = os.getenv("FIREBOLT_LLM_LOCATION")

        # Skip if required environment variables are not set
        if not all([client_id, client_secret, engine_name, database, account_name]):
            pytest.skip("Firebolt credentials not provided in environment variables")

        # Configure column_map to include common metadata columns for test framework
        # This allows the test framework to use arbitrary metadata fields
        config = FireboltSettings(
            id=client_id,
            secret=client_secret,
            engine_name=engine_name,
            database=database,
            account_name=account_name,
            table=table,
            index=index,
            metric=metric,
            llm_location=llm_location,
            embedding_model="amazon.titan-embed-text-v2:0",
            column_map={
                "id": "id",
                "document": "document",
                "embedding": "embedding",
                "metadata": ["some_other_field", "file_name", "page_number", "source", "title", "author"]
            }
        )

        vs = Firebolt(
            config=config,
            embeddings=embedding_openai,
            use_sql_embeddings=False,  # Use client-side embeddings for testing
        )
        # Store vectorstore on test instance for retriever_constructor property
        self.vectorstore = vs
        self._cached_vectorstore = vs
        return vs

    @pytest.fixture
    def retriever(
        self, vectorstore: Firebolt
    ) -> FireboltRetriever:
        """Create a FireboltRetriever instance for testing."""
        # Ensure vectorstore has some test documents for the retriever
        # Check if database is empty by trying to get one document
        try:
            test_results = vectorstore.similarity_search("test", k=1)
            if not test_results:
                # Database is empty, add default test documents
                from langchain_core.documents import Document
                test_docs = [
                    Document(page_content="test query", metadata={}),
                    Document(page_content="test document 1", metadata={}),
                    Document(page_content="test document 2", metadata={}),
                    Document(page_content="test document 3", metadata={}),
                ]
                vectorstore.add_documents(test_docs)
        except Exception:
            # If similarity_search fails, try adding documents anyway
            try:
                from langchain_core.documents import Document
                test_docs = [
                    Document(page_content="test query", metadata={}),
                    Document(page_content="test document 1", metadata={}),
                    Document(page_content="test document 2", metadata={}),
                    Document(page_content="test document 3", metadata={}),
                ]
                vectorstore.add_documents(test_docs)
            except Exception:
                # If adding documents fails, continue anyway
                pass
        
        return FireboltRetriever(
            vector_store=vectorstore,
            search_kwargs={"k": 4}
        )

    @pytest.fixture
    def embedding_openai(self) -> FakeEmbeddings:
        """Return a fake embeddings model for testing."""
        return FakeEmbeddings(size=256)

    @property
    def retriever_constructor(self):
        """Return a constructor function for creating retrievers."""
        def _construct(**kwargs):
            # Try to get vectorstore from multiple sources
            vs = None
            
            # First, try from test instance (set by fixture)
            vs = getattr(self, 'vectorstore', None)
            if vs is None:
                vs = getattr(self, '_cached_vectorstore', None)
            
            # If still None, try to get from pytest request
            if vs is None and hasattr(self, '_pytest_request'):
                try:
                    vs = self._pytest_request.getfixturevalue('vectorstore')
                except:
                    pass
            
            # Check if vs is actually a Firebolt instance
            from langchain_firebolt import Firebolt
            if vs is not None and not isinstance(vs, Firebolt):
                # If it's a fixture function, try to evaluate it via request
                if hasattr(self, '_pytest_request'):
                    try:
                        # getfixturevalue should return the actual value, not the function
                        vs = self._pytest_request.getfixturevalue('vectorstore')
                        # Double-check it's now a Firebolt instance
                        if not isinstance(vs, Firebolt):
                            # If still not, the fixture might not have run yet
                            # Try to get it from cached value
                            vs = getattr(self, '_cached_vectorstore', None)
                    except Exception as e:
                        # Fallback to cached value
                        vs = getattr(self, '_cached_vectorstore', None)
            
            if vs is None or not isinstance(vs, Firebolt):
                raise ValueError(f"vectorstore not available or invalid. Got: {type(vs)}. Ensure vectorstore fixture is used.")
            
            # Ensure vectorstore has some test documents for the retriever
            # The test framework might add documents, but we'll add some defaults if empty
            # Check if database is empty by trying to get one document
            try:
                test_results = vs.similarity_search("test", k=1)
                if not test_results:
                    # Database is empty, add default test documents
                    from langchain_core.documents import Document
                    test_docs = [
                        Document(page_content="test query", metadata={}),
                        Document(page_content="test document 1", metadata={}),
                        Document(page_content="test document 2", metadata={}),
                        Document(page_content="test document 3", metadata={}),
                    ]
                    vs.add_documents(test_docs)
            except Exception:
                # If similarity_search fails, try adding documents anyway
                # This handles cases where the table might be empty
                try:
                    from langchain_core.documents import Document
                    test_docs = [
                        Document(page_content="test query", metadata={}),
                        Document(page_content="test document 1", metadata={}),
                        Document(page_content="test document 2", metadata={}),
                        Document(page_content="test document 3", metadata={}),
                    ]
                    vs.add_documents(test_docs)
                except Exception as e:
                    # If adding documents fails, log but continue
                    # The test might handle empty database differently
                    import logging
                    logging.debug(f"Could not add default test documents: {e}")
            
            # Extract search_kwargs, defaulting to k from kwargs or 4
            search_kwargs = kwargs.pop("search_kwargs", {})
            # If k is in kwargs, add it to search_kwargs
            if "k" in kwargs:
                search_kwargs["k"] = kwargs.pop("k")
            if not search_kwargs:
                search_kwargs = {"k": 4}
            
            # Create retriever with vectorstore and any additional kwargs
            return FireboltRetriever(vector_store=vs, search_kwargs=search_kwargs, **kwargs)
        
        return _construct

    @property
    def supports_async(self) -> bool:
        """FireboltRetriever supports async methods."""
        return True

    @property
    def supports_scores(self) -> bool:
        """FireboltRetriever supports scores."""
        return True
    
    @property
    def retriever_query_example(self) -> str:
        """Return an example query for testing."""
        return "test query"
    
    @property
    def num_results_arg_name(self) -> str:
        """Return the name of the argument for number of results."""
        return "k"
    
    @property
    def retriever_constructor_params(self) -> Dict[str, Any]:
        """Return default parameters for retriever constructor."""
        return {}

