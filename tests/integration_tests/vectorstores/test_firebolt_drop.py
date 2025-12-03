"""Integration tests for Firebolt vector store drop method."""

import os
import pytest
from dotenv import load_dotenv

load_dotenv()

from langchain_core.embeddings import FakeEmbeddings
from langchain_firebolt import Firebolt, FireboltSettings


@pytest.fixture
def vectorstore(firebolt_table_setup: dict) -> Firebolt:
    """Create a Firebolt vector store instance for testing."""
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
            "metadata": ["some_other_field", "file_name", "page_number", "source", "title", "author"]
        }
    )

    return Firebolt(
        config=config,
        embeddings=FakeEmbeddings(size=256),
        use_sql_embeddings=True,
    )


def test_drop_raises_not_implemented_by_default(vectorstore):
    """Test that drop() raises NotImplementedError by default."""
    with pytest.raises(NotImplementedError, match="drop.*is not implemented by default"):
        vectorstore.drop()


def test_drop_with_drop_table_false(vectorstore):
    """Test that drop() raises NotImplementedError when drop_table=False."""
    with pytest.raises(NotImplementedError, match="drop.*is not implemented by default"):
        vectorstore.drop(drop_table=False)


def test_drop_with_drop_table_true(vectorstore):
    """Test that drop() successfully drops the index and table when drop_table=True."""
    # Add some documents first
    from langchain_core.documents import Document
    docs = [
        Document(page_content="test document 1", metadata={"file_name": "test1.pdf"}),
        Document(page_content="test document 2", metadata={"file_name": "test2.pdf"}),
    ]
    vectorstore.add_documents(docs)
    
    # Verify documents were added
    results = vectorstore.similarity_search("test", k=2)
    assert len(results) == 2
    
    # Drop the table and index
    vectorstore.drop(drop_table=True)
    
    # Verify the table was dropped by trying to query it (should fail)
    # The table should no longer exist, so any operation should fail
    # We can verify this by checking that the connection still works but the table doesn't exist
    from firebolt.db import connect
    from firebolt.client.auth import ClientCredentials
    
    client_id = os.getenv("FIREBOLT_CLIENT_ID")
    client_secret = os.getenv("FIREBOLT_CLIENT_SECRET")
    engine_name = os.getenv("FIREBOLT_ENGINE")
    database = os.getenv("FIREBOLT_DB")
    account_name = os.getenv("FIREBOLT_ACCOUNT")
    api_endpoint = os.getenv("FIREBOLT_API_ENDPOINT")
    
    auth = ClientCredentials(client_id=client_id, client_secret=client_secret)
    connection_params = {
        "engine_name": engine_name,
        "database": database,
        "account_name": account_name,
        "auth": auth
    }
    if api_endpoint:
        if "staging" in api_endpoint.lower():
            connection_params["api_endpoint"] = "https://api.staging.firebolt.io"
        else:
            connection_params["api_endpoint"] = api_endpoint
    
    conn = connect(**connection_params)
    cursor = conn.cursor()
    
    try:
        # Try to query the table - it should not exist
        cursor.execute(f"SELECT COUNT(*) FROM {vectorstore.config.table}")
        # If we get here, the table still exists (which would be unexpected)
        pytest.fail("Table should have been dropped but still exists")
    except Exception as e:
        # Expected: table doesn't exist
        error_msg = str(e).lower()
        assert "does not exist" in error_msg or "not found" in error_msg or "unknown" in error_msg
    finally:
        cursor.close()
        conn.close()

