"""Integration tests for Firebolt vector store."""

import os
import pytest
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests

from langchain_firebolt import Firebolt, FireboltSettings


class TestFireboltIntegration(VectorStoreIntegrationTests):
    """Integration tests for Firebolt vector store.
    
    Note: The test framework uses metadata['id'] which conflicts with our column_map
    design where 'id' is the primary key column. We enforce that 'id' cannot be in
    metadata_cols, so tests that expect metadata['id'] to differ from Document.id
    will fail. These tests are marked as xfail.
    """
    
    @pytest.mark.xfail(
        reason="This test uses Document(id='foo', metadata={'id': 1}) where Document.id "
               "differs from metadata['id']. Our implementation enforces that 'id' cannot "
               "be in metadata_cols, so metadata['id'] is set from the primary key column."
    )
    def test_add_documents_with_existing_ids(self, vectorstore):
        """Override to mark as xfail."""
        return super().test_add_documents_with_existing_ids(vectorstore)
    
    @pytest.mark.xfail(
        reason="This test uses Document(id='foo', metadata={'id': 1}) where Document.id "
               "differs from metadata['id']. Our implementation enforces that 'id' cannot "
               "be in metadata_cols, so metadata['id'] is set from the primary key column."
    )
    async def test_add_documents_with_existing_ids_async(self, vectorstore):
        """Override to mark as xfail."""
        return await super().test_add_documents_with_existing_ids_async(vectorstore)

    @pytest.fixture
    def vectorstore_cls(self) -> type:
        """Return the Firebolt vector store class."""
        return Firebolt

    @pytest.fixture
    def vectorstore(
        self, 
        vectorstore_cls: type, 
        embedding_openai: FakeEmbeddings,
        firebolt_table_setup: dict
    ) -> Firebolt:
        """Create a Firebolt vector store instance for testing.
        
        Args:
            vectorstore_cls: The Firebolt vector store class.
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

        return vectorstore_cls(
            config=config,
            embeddings=embedding_openai,
            use_sql_embeddings=True,  # Use SQL embeddings (AI_EMBED_TEXT) for deterministic results
        )

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

