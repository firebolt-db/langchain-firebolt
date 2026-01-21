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


class TestFireboltMetadataFilterIntegration:
    """Integration tests for metadata filtering with the k_multiplier feature.
    
    These tests verify that:
    1. Results never exceed the requested k even when the multiplier fetches more candidates
    2. Results can be fewer than k when the filter is too restrictive
    """

    @pytest.fixture
    def vectorstore_with_data(
        self,
        firebolt_table_setup: dict
    ) -> Firebolt:
        """Create a Firebolt vector store with test data for metadata filtering tests.
        
        Creates a table with 20 documents, each with a 'category' metadata field:
        - 10 documents with category='A'
        - 5 documents with category='B'
        - 5 documents with category='C'
        """
        import os
        import uuid
        
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
                "metadata": ["category", "seq_num"]
            }
        )

        vector_store = Firebolt(
            config=config,
            use_sql_embeddings=True,
        )
        
        # Create test documents with varied categories
        documents = []
        
        # Category A: 10 documents about technology
        for i in range(10):
            documents.append(Document(
                page_content=f"Technology article about computing and software number {i}",
                metadata={"category": "A", "seq_num": i},
                id=str(uuid.uuid4())
            ))
        
        # Category B: 5 documents about science
        for i in range(5):
            documents.append(Document(
                page_content=f"Science article about physics and chemistry number {i}",
                metadata={"category": "B", "seq_num": i + 10},
                id=str(uuid.uuid4())
            ))
        
        # Category C: 5 documents about art
        for i in range(5):
            documents.append(Document(
                page_content=f"Art article about painting and sculpture number {i}",
                metadata={"category": "C", "seq_num": i + 15},
                id=str(uuid.uuid4())
            ))
        
        # Add documents to vector store
        vector_store.add_documents(documents)
        
        return vector_store

    def test_metadata_filter_returns_at_most_k_results(
        self, vectorstore_with_data: Firebolt
    ):
        """Test that metadata filtering with use_index=True returns at most k results.
        
        Even though the metadata_filter_k_multiplier causes more candidates to be 
        fetched from the index (k * multiplier), the final result should be limited
        to at most k documents.
        
        Scenario:
        - Request k=3 results with filter category='A' (10 matching docs exist)
        - With default multiplier of 10, vector_search fetches 30 candidates
        - Final result should be exactly 3 documents (or fewer)
        """
        vector_store = vectorstore_with_data
        
        # Search with metadata filter for category A (10 docs available)
        # Request k=3, with multiplier=10, vector_search will fetch 30 candidates
        results = vector_store.similarity_search(
            query="technology software computing",
            k=3,
            filter={"category": "A"},
            use_index=True,
            metadata_filter_k_multiplier=10  # Default, but explicit for clarity
        )
        
        # Verify we get at most k results
        assert len(results) <= 3, (
            f"Expected at most 3 results, got {len(results)}. "
            "The metadata_filter_k_multiplier should not increase the final result count."
        )
        
        # Verify all results match the filter
        for doc in results:
            assert doc.metadata.get("category") == "A", (
                f"Document should have category='A', got {doc.metadata.get('category')}"
            )

    def test_metadata_filter_returns_fewer_than_k_when_restrictive(
        self, vectorstore_with_data: Firebolt
    ):
        """Test that a restrictive metadata filter can return fewer than k results.
        
        Scenario:
        - Request k=10 results with filter category='C' (only 5 matching docs exist)
        - Result should be exactly 5 documents (fewer than requested k)
        """
        vector_store = vectorstore_with_data
        
        # Search with restrictive filter for category C (only 5 docs available)
        # Request k=10, but only 5 match the filter
        results = vector_store.similarity_search(
            query="art painting sculpture",
            k=10,
            filter={"category": "C"},
            use_index=True,
            metadata_filter_k_multiplier=10
        )
        
        # Verify we get fewer than k results (only 5 docs match)
        assert len(results) <= 5, (
            f"Expected at most 5 results (category C has only 5 docs), got {len(results)}"
        )
        
        # Verify all results match the filter
        for doc in results:
            assert doc.metadata.get("category") == "C", (
                f"Document should have category='C', got {doc.metadata.get('category')}"
            )

    def test_metadata_filter_with_no_matching_results(
        self, vectorstore_with_data: Firebolt
    ):
        """Test that a filter with no matches returns empty results.
        
        Scenario:
        - Request results with filter category='X' (no matching docs exist)
        - Result should be empty
        """
        vector_store = vectorstore_with_data
        
        # Search with filter for non-existent category
        results = vector_store.similarity_search(
            query="any query text",
            k=5,
            filter={"category": "X"},  # No documents have category X
            use_index=True,
            metadata_filter_k_multiplier=10
        )
        
        # Verify we get no results
        assert len(results) == 0, (
            f"Expected 0 results for non-existent category, got {len(results)}"
        )

    def test_metadata_filter_with_score_returns_at_most_k(
        self, vectorstore_with_data: Firebolt
    ):
        """Test that similarity_search_with_score also respects the k limit with filters.
        
        Scenario:
        - Request k=3 results with score and filter category='A' (10 matching docs exist)
        - Result should be at most 3 (document, score) tuples
        """
        vector_store = vectorstore_with_data
        
        # Search with score and metadata filter
        results = vector_store.similarity_search_with_score(
            query="technology software computing",
            k=3,
            filter={"category": "A"},
            use_index=True,
            metadata_filter_k_multiplier=10
        )
        
        # Verify we get at most k results
        assert len(results) <= 3, (
            f"Expected at most 3 results, got {len(results)}"
        )
        
        # Verify all results are tuples with (Document, score)
        for doc, score in results:
            assert isinstance(doc, Document), "First element should be a Document"
            assert isinstance(score, float), "Second element should be a float score"
            assert doc.metadata.get("category") == "A", (
                f"Document should have category='A', got {doc.metadata.get('category')}"
            )

    def test_metadata_filter_with_higher_multiplier(
        self, vectorstore_with_data: Firebolt
    ):
        """Test that increasing the multiplier can improve recall with restrictive filters.
        
        Scenario:
        - Request k=5 with filter category='B' (5 matching docs)
        - With a higher multiplier, we should get all 5 matching docs
        """
        vector_store = vectorstore_with_data
        
        # Search with higher multiplier to ensure we get all matching results
        results = vector_store.similarity_search(
            query="science physics chemistry",
            k=5,
            filter={"category": "B"},
            use_index=True,
            metadata_filter_k_multiplier=20  # Higher multiplier for better recall
        )
        
        # We should get exactly 5 or fewer results
        assert len(results) <= 5, f"Expected at most 5 results, got {len(results)}"
        
        # Verify all results match the filter
        for doc in results:
            assert doc.metadata.get("category") == "B", (
                f"Document should have category='B', got {doc.metadata.get('category')}"
            )
