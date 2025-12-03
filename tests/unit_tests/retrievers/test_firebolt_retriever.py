"""Unit tests for Firebolt retriever."""

import pytest
from unittest.mock import patch
from typing import Any, Dict, List
import uuid

from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_tests.integration_tests.retrievers import RetrieversIntegrationTests

from langchain_firebolt import (
    Firebolt,
    FireboltRetriever,
    FireboltSettings,
)


class TestFireboltRetrieverUnit(RetrieversIntegrationTests):
    """Unit tests for Firebolt retriever."""

    @pytest.fixture
    def retriever_cls(self) -> type:
        """Return the FireboltRetriever class."""
        return FireboltRetriever

    @pytest.fixture(autouse=True)
    def _mock_firebolt_connection(self, mock_firebolt_connection, request):
        """Auto-use fixture to mock Firebolt connection for all tests."""
        # Store request for retriever_constructor property
        self._pytest_request = request
        with patch("firebolt.db.connect", return_value=mock_firebolt_connection):
            yield mock_firebolt_connection

    @pytest.fixture
    def vectorstore(
        self, embedding_openai: FakeEmbeddings, _mock_firebolt_connection, mock_firebolt_db
    ) -> Firebolt:
        """Create a Firebolt vector store instance for testing."""
        mock_connection = _mock_firebolt_connection
        
        config = FireboltSettings(
            id="test_id",
            secret="test_secret",
            engine_name="test_engine",
            database="test_db",
            account_name="test_account",
            table="test_table",
            index="test_index",
            llm_location="test_location",
            embedding_model="amazon.titan-embed-text-v2:0",
        )

        # Wrap the embeddings model to use cached embeddings from mock database
        # This ensures deterministic behavior - same text always produces same embedding
        from tests.conftest import CachedFakeEmbeddings
        cached_embeddings = CachedFakeEmbeddings(embedding_openai, mock_firebolt_db.text_to_embedding)
        
        vs = Firebolt(
            config=config,
            embeddings=cached_embeddings,
            use_sql_embeddings=False,
        )
        # Keep the mock connection and database accessible
        vs._mock_connection = mock_connection
        vs._mock_db = mock_firebolt_db
        
        # Intercept add_texts to populate mock database
        original_add_texts = vs.add_texts
        def add_texts_with_mock(*args, **kwargs):
            # Get texts and other params before calling original
            texts = list(args[0]) if args else kwargs.get('texts', [])
            ids = kwargs.get('ids')
            metadatas = kwargs.get('metadatas')
            embeddings = kwargs.get('embeddings')
            
            # Store original metadata types before converting IDs to strings
            original_metadatas = metadatas.copy() if metadatas else None
            
            # Ensure IDs are strings if provided (for Firebolt, but preserve in metadata)
            if ids:
                ids = [str(id_val) for id_val in ids]
                kwargs['ids'] = ids
            
            result = original_add_texts(*args, **kwargs)
            # After add_texts, populate mock_db with the added documents
            # Ensure result IDs are strings
            result_ids = [str(id_val) for id_val in result] if result else []
            
            if not ids:
                ids = result_ids  # Use returned IDs
            
            for i, text in enumerate(texts):
                doc_id = str(ids[i]) if ids and i < len(ids) else str(uuid.uuid4())
                # Use original metadata to preserve types (e.g., int IDs)
                if original_metadatas and i < len(original_metadatas):
                    metadata = original_metadatas[i].copy() if original_metadatas[i] else {}
                elif metadatas and i < len(metadatas):
                    metadata = metadatas[i].copy() if metadatas[i] else {}
                else:
                    metadata = {}
                
                # Ensure the id in metadata matches doc_id (as string for storage)
                # But preserve original type if it was in metadata
                if 'id' in metadata:
                    # Keep original type, but also store string version for doc_id
                    pass  # Keep original metadata['id'] type
                else:
                    metadata['id'] = doc_id
                
                embedding = embeddings[i] if embeddings and i < len(embeddings) else cached_embeddings.embed_query(text)
                
                mock_firebolt_db.add_document(doc_id, text, embedding, metadata)
            
            return result_ids
        
        vs.add_texts = add_texts_with_mock
        # Store vectorstore on test instance for retriever_constructor property
        # Store both as vectorstore (expected by test framework) and _cached_vectorstore (fallback)
        self.vectorstore = vs
        self._cached_vectorstore = vs
        return vs

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
            if hasattr(vs, '_mock_db'):
                mock_db = vs._mock_db
                if not mock_db.documents:
                    # Add some default test documents
                    from langchain_core.documents import Document
                    from langchain_core.embeddings import FakeEmbeddings
                    embedding_model = getattr(vs, 'embeddings', FakeEmbeddings(size=256))
                    test_docs = [
                        Document(page_content="test query", metadata={}),
                        Document(page_content="test document 1", metadata={}),
                        Document(page_content="test document 2", metadata={}),
                        Document(page_content="test document 3", metadata={}),
                    ]
                    vs.add_documents(test_docs)
            
            # Extract search_kwargs, defaulting to k from kwargs or 4
            search_kwargs = kwargs.pop("search_kwargs", {})
            # If k is in kwargs, add it to search_kwargs
            if "k" in kwargs:
                search_kwargs["k"] = kwargs.pop("k")
            if not search_kwargs:
                search_kwargs = {"k": 4}
            return FireboltRetriever(
                vector_store=vs,
                search_kwargs=search_kwargs,
                **kwargs
            )
        return _construct
    
    @property
    def retriever_constructor_params(self):
        """Return default parameters for retriever constructor."""
        return {}
    
    @property
    def retriever_query_example(self) -> str:
        """Return an example query for testing."""
        return "test query"
    
    @property
    def num_results_arg_name(self) -> str:
        """Return the name of the argument for number of results."""
        return "k"

    @pytest.fixture
    def retriever(
        self, vectorstore: Firebolt, embedding_openai: FakeEmbeddings
    ) -> FireboltRetriever:
        """Create a FireboltRetriever instance for testing."""
        # Add some default test documents to the vectorstore
        test_docs = [
            Document(page_content="test document 1", metadata={}),
            Document(page_content="test document 2", metadata={}),
            Document(page_content="test document 3", metadata={}),
        ]
        vectorstore.add_documents(test_docs)
        
        return FireboltRetriever(
            vector_store=vectorstore,
            search_kwargs={"k": 4}
        )

    @pytest.fixture
    def embedding_openai(self) -> FakeEmbeddings:
        """Return a fake embeddings model for testing."""
        return FakeEmbeddings(size=256)

    @property
    def supports_async(self) -> bool:
        """FireboltRetriever supports async methods."""
        return True

    @property
    def supports_scores(self) -> bool:
        """FireboltRetriever supports scores."""
        return True

