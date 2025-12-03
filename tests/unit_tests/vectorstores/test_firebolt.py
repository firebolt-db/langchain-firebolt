"""Unit tests for Firebolt vector store."""

import pytest
from unittest.mock import patch
from typing import Any, Dict, List
import uuid

from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests

from langchain_firebolt import Firebolt, FireboltSettings

class TestFireboltUnit(VectorStoreIntegrationTests):
    """Unit tests for Firebolt vector store."""
    
    @pytest.mark.xfail(
        reason="This test uses Document(id='foo', metadata={'id': 1}) where Document.id "
               "differs from metadata['id']. Our implementation enforces that 'id' cannot "
               "also be in metadata columns because all columns are columns in the Firebolt table."
    )
    def test_add_documents_with_existing_ids(self, vectorstore):
        """Override to mark as xfail."""
        return super().test_add_documents_with_existing_ids(vectorstore)
    
    @pytest.mark.xfail(
        reason="This test uses Document(id='foo', metadata={'id': 1}) where Document.id "
               "differs from metadata['id']. Our implementation enforces that 'id' cannot "
               "also be in metadata columns because all columns are columns in the Firebolt table."
    )
    async def test_add_documents_with_existing_ids_async(self, vectorstore):
        """Override to mark as xfail."""
        return await super().test_add_documents_with_existing_ids_async(vectorstore)

    @pytest.fixture
    def vectorstore_cls(self) -> type:
        """Return the Firebolt vector store class."""
        return Firebolt

    @pytest.fixture(autouse=True)
    def _mock_firebolt_connection(self, mock_firebolt_connection):
        """Auto-use fixture to mock Firebolt connection for all tests."""
        with patch("firebolt.db.connect", return_value=mock_firebolt_connection):
            yield mock_firebolt_connection

    @pytest.fixture
    def vectorstore(
        self, vectorstore_cls: type, embedding_openai: FakeEmbeddings, _mock_firebolt_connection, mock_firebolt_db
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
        
        vs = vectorstore_cls(
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
            original_ids = kwargs.get('ids')
            metadatas = kwargs.get('metadatas')
            embeddings = kwargs.get('embeddings')
            
            # Store original IDs and metadata types before converting
            original_metadatas = metadatas.copy() if metadatas else None
            # Preserve original IDs as-is (could be strings, ints, etc.)
            original_ids_list = list(original_ids) if original_ids else None
            
            # Convert IDs to strings for Firebolt storage, but preserve originals for return
            if original_ids:
                ids_for_firebolt = [str(id_val) for id_val in original_ids]
                kwargs['ids'] = ids_for_firebolt
            else:
                ids_for_firebolt = None
            
            result = original_add_texts(*args, **kwargs)
            # After add_texts, populate mock_db with the added documents
            # The result from Firebolt will be string IDs, but we should return original format
            result_ids = list(result) if result else []
            
            # Use original_ids if provided, otherwise use result_ids
            ids_to_use = original_ids_list if original_ids_list else result_ids
            
            for i, text in enumerate(texts):
                # Use string ID for storage in mock database
                doc_id_str = str(ids_to_use[i]) if ids_to_use and i < len(ids_to_use) else str(uuid.uuid4())
                # But preserve original ID format for return value
                original_id = ids_to_use[i] if ids_to_use and i < len(ids_to_use) else doc_id_str
                
                # Use original metadata to preserve types (e.g., int IDs) and all fields
                if original_metadatas and i < len(original_metadatas):
                    metadata = original_metadatas[i].copy() if original_metadatas[i] else {}
                elif metadatas and i < len(metadatas):
                    metadata = metadatas[i].copy() if metadatas[i] else {}
                else:
                    metadata = {}
                
                # Set the id in metadata to the original ID (preserving type)
                # But don't overwrite if it's already set in metadata
                if 'id' not in metadata:
                    metadata['id'] = original_id
                # If metadata has 'id' but it's different from original_id, preserve the metadata['id']
                # This handles cases where Document.id is set separately from metadata['id']
                
                embedding = embeddings[i] if embeddings and i < len(embeddings) else cached_embeddings.embed_query(text)
                
                # Add or update document in mock database (upsert behavior)
                # Pass the full metadata including all fields
                mock_firebolt_db.add_document(doc_id_str, text, embedding, metadata)
            
            # Return original IDs if provided, otherwise return result_ids
            return original_ids_list if original_ids_list else result_ids
        
        vs.add_texts = add_texts_with_mock
        return vs

    @pytest.fixture
    def embedding_openai(self) -> FakeEmbeddings:
        """Return a fake embeddings model for testing."""
        return FakeEmbeddings(size=256)

    @property
    def supports_add_texts(self) -> bool:
        """Firebolt supports add_texts."""
        return True

    @property
    def supports_add_documents(self) -> bool:
        """Firebolt supports add_documents."""
        return True

    @property
    def supports_similarity_search(self) -> bool:
        """Firebolt supports similarity_search."""
        return True

    @property
    def supports_similarity_search_by_vector(self) -> bool:
        """Firebolt supports similarity_search_by_vector."""
        return True

    @property
    def supports_similarity_search_with_score(self) -> bool:
        """Firebolt supports similarity_search_with_score."""
        return True

    @property
    def supports_delete(self) -> bool:
        """Firebolt supports delete."""
        return True

    @property
    def supports_get_by_ids(self) -> bool:
        """Firebolt supports get_by_ids."""
        return True
    
    @property
    def has_get_by_ids(self) -> bool:
        """Firebolt supports get_by_ids."""
        return True

    @property
    def supports_metadata(self) -> bool:
        """Firebolt supports metadata."""
        return True

    @property
    def supports_filtering(self) -> bool:
        """Firebolt supports filtering."""
        return True

    @pytest.fixture
    def texts(self) -> List[str]:
        """Return sample texts for testing."""
        return ["foo", "bar", "baz"]

    @pytest.fixture
    def metadatas(self) -> List[Dict[str, Any]]:
        """Return sample metadatas for testing."""
        return [{"key": f"value_{i}"} for i in range(3)]

