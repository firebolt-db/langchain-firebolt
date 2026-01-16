# LangChain Firebolt Vector Store

A LangChain vector store integration for Firebolt, enabling efficient similarity search and document management using Firebolt's vector search capabilities.

## Table of Contents

- [Installation](#installation)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Environment Variables](#environment-variables)
- [Best Practices](#best-practices)

## Installation

```bash
pip install langchain-firebolt
```

## Prerequisites

### 1. Firebolt Account Setup

You need:
- A Firebolt account ([Sign up for a free trial](https://go.firebolt.io/signup))
- An engine running in your account
- A database created in your account
- Client credentials (client ID and secret, [see how to create a service account](https://docs.firebolt.io/guides/managing-your-organization/service-accounts#create-a-service-account))

### 2. Create LOCATION Object for LLM API (Optional)

**Note:** This step is only required when using server-side embedding calculation (`use_sql_embeddings=True`, which is the default). If you're using client-side embeddings (`use_sql_embeddings=False`), you can skip this step.

The Firebolt vector store can use Firebolt's `AI_EMBED_TEXT` SQL function to generate embeddings server-side. When using this feature, you need to create a LOCATION object in Firebolt that points to your LLM service (e.g., Amazon Bedrock).

**Example: Creating a LOCATION for Amazon Bedrock**

```sql
CREATE LOCATION llm_api WITH
  SOURCE = AMAZON_BEDROCK
  CREDENTIALS = 
    (
    AWS_ACCESS_KEY_ID='your_access_key'
    AWS_SECRET_ACCESS_KEY='your_secret_key'
    );
```

For more details, see the [Firebolt documentation](https://docs.firebolt.io/reference-sql/commands/data-definition/create-location-bedrock#create-location-amazon-bedrock).

### 3. Create Table and Vector Index

The table and vector index will be automatically created when you instantiate the `Firebolt` vector store if they don't already exist. However, you can also create them manually beforehand using the SQL commands below.

**Automatic Creation:**
- If the table doesn't exist, it will be created automatically with the required structure
- If the table exists but the index doesn't exist, the index will be created automatically
- The index name will be auto-generated as `{table_name}_index` if not specified in your configuration

**Manual Creation (Optional):**

Create a table with the following structure:

```sql
CREATE TABLE IF NOT EXISTS documents (
    id TEXT,
    document TEXT,
    embedding ARRAY(DOUBLE PRECISION NOT NULL) NOT NULL,
    -- Add your metadata columns here
    file_name TEXT,
    page_number INTEGER,
    source TEXT
) PRIMARY INDEX id;
```

Create a vector search index:

```sql
CREATE INDEX documents_index
ON documents
USING HNSW(embedding vector_cosine_ops) WITH (dimension = 256);
```

**Supported metrics:**
- `vector_cosine_ops` (default) - Cosine similarity
- `vector_ip_ops` - Inner product
- `vector_l2sq_ops` - L2 squared distance

## Quick Start

```python
from langchain_firebolt import Firebolt, FireboltSettings
from langchain_core.documents import Document

# Configure Firebolt settings
settings = FireboltSettings(
    id="your_client_id",
    secret="your_client_secret",
    engine_name="your_engine",
    database="my_database",
    account_name="your_account",
    table="documents",
    index="documents_index",  # Optional: auto-detected if not provided
    llm_location="llm_api",  # Required for server-side embeddings (use_sql_embeddings=True)
    embedding_model="amazon.titan-embed-text-v2:0",  # Required for server-side embeddings
    embedding_dimension=256,
    metric="vector_cosine_ops",  # Optional: defaults to vector_cosine_ops
)

# Create vector store instance
vector_store = Firebolt(config=settings)

# Add documents
documents = [
    Document(page_content="The quick brown fox jumps over the lazy dog", metadata={"file_name": "doc1.txt"}),
    Document(page_content="Python is a programming language", metadata={"file_name": "doc2.txt"}),
]
vector_store.add_documents(documents)

# Search
results = vector_store.similarity_search("programming", k=2)
for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
```

## Configuration

### FireboltSettings

The `FireboltSettings` class configures the connection and behavior of the vector store.

#### Required Parameters

- `id` (str): Firebolt client ID
- `secret` (str): Firebolt client secret
- `engine_name` (str): Name of the Firebolt engine
- `database` (str): Name of the database
- `account_name` (str): Firebolt account name
- `table` (str): Name of the table containing vectors
- `embedding_model` (str): Embedding model identifier (e.g., "amazon.titan-embed-text-v2:0")

#### Optional Parameters

- `index` (str, optional): Vector index name. If not provided, will be auto-detected from the database.
- `llm_location` (str, optional): Name of the LOCATION object in Firebolt. Required when `use_sql_embeddings=True`.
- `embedding_dimension` (int): Dimension of embeddings. Defaults to 256.
- `batch_size` (int): Batch size for MERGE operations. Defaults to 32.
- `metric` (str): Similarity metric. Options: `"vector_cosine_ops"` (default), `"vector_ip_ops"`, `"vector_l2sq_ops"`.
- `api_endpoint` (str, optional): Custom API endpoint. Defaults to Firebolt's cloud API.
- `column_map` (dict): Mapping of LangChain semantics to table columns. Defaults to:
  ```python
  {
      "id": "id",
      "document": "document",
      "embedding": "embedding",
      "metadata": []  # List of metadata column names
  }
  ```

### Firebolt Constructor

```python
Firebolt(
    config: Optional[FireboltSettings] = None,
    embeddings: Optional[Embeddings] = None,
    use_sql_embeddings: bool = True,
    **kwargs
)
```

**Parameters:**
- `config` (FireboltSettings, optional): Configuration object. If None, will use environment variables.
- `embeddings` (Embeddings, optional): Embeddings model. Required if `use_sql_embeddings=False`.
- `use_sql_embeddings` (bool): Whether to use SQL-based embeddings (`AI_EMBED_TEXT`). Defaults to `True`.

## Usage Examples

### Adding Documents

#### Using `add_documents()`

```python
from langchain_core.documents import Document

documents = [
    Document(
        page_content="Machine learning is a subset of artificial intelligence",
        metadata={"file_name": "ml_intro.pdf", "page_number": 1}
    ),
    Document(
        page_content="Deep learning uses neural networks",
        metadata={"file_name": "dl_basics.pdf", "page_number": 1}
    ),
]

vector_store.add_documents(documents)
```

#### Using `add_texts()`

```python
texts = ["First document", "Second document"]
metadatas = [{"source": "doc1"}, {"source": "doc2"}]
ids = ["id1", "id2"]

vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
```

#### Using Precomputed Embeddings

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vector_store = Firebolt(
    config=settings,
    embeddings=embeddings,
    use_sql_embeddings=False  # Use client-side embeddings
)

# Add documents with precomputed embeddings
vector_store.add_documents(documents)
```

#### Batch Processing

```python
# Process documents in batches
vector_store.add_documents(documents, batch_size=64)
```

### Searching Documents

#### Basic Similarity Search

```python
results = vector_store.similarity_search("machine learning", k=5)
for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
```

#### Search with Scores

```python
results = vector_store.similarity_search_with_score("neural networks", k=3)
for doc, score in results:
    print(f"Score: {score:.4f}")
    print(f"Content: {doc.page_content}")
```

**Note:** Score interpretation depends on the metric:
- For `vector_cosine_ops` and `vector_l2sq_ops`: Lower scores indicate higher similarity
- For `vector_ip_ops`: Uses `1 - VECTOR_INNER_PRODUCT`, so lower scores indicate higher similarity

#### Search with Filters

```python
# Filter by metadata
results = vector_store.similarity_search(
    query="machine learning",
    k=5,
    filter={"file_name": "ml_intro.pdf", "page_number": 1}
)

# Filter with multiple values (IN clause)
results = vector_store.similarity_search(
    query="neural networks",
    k=5,
    filter={"file_name": ["doc1.pdf", "doc2.pdf"]}
)

# Filter for NULL values
results = vector_store.similarity_search(
    query="test",
    k=5,
    filter={"source": None}
)
```

#### Index vs Brute Force Search

The `use_index` parameter controls whether to use the vector search index (fast approximate search) or brute force table scan (exact search):

- `use_index=True` (default): Uses the `vector_search` TVF for fast approximate nearest neighbor search
- `use_index=False`: Scans the entire table, calculates distances, and sorts results (exact search)

```python
# Use vector search index (default behavior)
results = vector_store.similarity_search(
    query="machine learning",
    k=5
)

# Uses vector search index by default
results = vector_store.similarity_search(
    query="machine learning",
    k=5
)

# Use brute force table scan (exact search)
results = vector_store.similarity_search(
    query="machine learning",
    k=5,
    use_index=False
)

# Brute force with filter
results = vector_store.similarity_search(
    query="machine learning",
    k=5,
    filter={"file_name": "doc.pdf"},
    use_index=False
)
```

The `use_index` parameter is available on all similarity search methods:
- `similarity_search()`
- `similarity_search_with_score()`
- `similarity_search_by_vector()`
- `similarity_search_with_score_by_vector()`

#### Metadata Filter K Multiplier

When using index-based search (`use_index=True`) with metadata filters, the vector search index returns approximate nearest neighbors *before* the filter is applied. This can result in fewer results than requested if many candidates are filtered out.

The `metadata_filter_k_multiplier` parameter (default: 10) addresses this by requesting more results from the index:

```python
# Request 5 results, but fetch 50 candidates from the index to ensure enough after filtering
results = vector_store.similarity_search(
    query="machine learning",
    k=5,
    filter={"category": "technical"},
    metadata_filter_k_multiplier=10  # default, fetches k*10 from index
)

# Increase multiplier for very selective filters
results = vector_store.similarity_search(
    query="machine learning",
    k=5,
    filter={"category": "rare_category"},
    metadata_filter_k_multiplier=50  # fetches k*50 from index
)

# Disable multiplier (only useful when you know most results will match)
results = vector_store.similarity_search(
    query="machine learning",
    k=5,
    filter={"category": "common_category"},
    metadata_filter_k_multiplier=1  # fetches exactly k from index
)
```

**Note:** The multiplier only affects index-based searches with filters. Brute force searches (`use_index=False`) always scan the entire table, so filtering doesn't reduce the result count.

#### Search by Vector

```python
# Get embedding first
query_embedding = vector_store._get_embedding("machine learning")

# Search using the vector
results = vector_store.similarity_search_by_vector(query_embedding, k=5)
```

### Retrieving Documents by ID

```python
# Get documents by their IDs
ids = ["id1", "id2", "id3"]
documents = vector_store.get_by_ids(ids)

for doc in documents:
    print(f"ID: {doc.metadata['id']}")
    print(f"Content: {doc.page_content}")
```

### Deleting Documents

#### Delete by IDs

```python
vector_store.delete(ids=["id1", "id2"])
```

#### Delete by Filter

```python
# Delete all documents from a specific file
vector_store.delete(filter={"file_name": "old_document.pdf"})
```

#### Delete All Documents

```python
# WARNING: This deletes all documents in the table
vector_store.delete(delete_all=True)
```

### Dropping Table and Index

```python
# WARNING: This permanently deletes the table and index
# Requires explicit confirmation
vector_store.drop(drop_table=True)
```

### Using as Retriever

```python
from langchain_firebolt import FireboltRetriever

# Create a retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5, "filter": {"source": "docs"}})

# Use in a chain
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

result = qa_chain.invoke({"query": "What is machine learning?"})
```

### Class Methods

#### `from_documents()`

```python
vector_store = Firebolt.from_documents(
    documents=documents,
    config=settings,
    use_sql_embeddings=True
)
```

#### `from_texts()`

```python
vector_store = Firebolt.from_texts(
    texts=["Text 1", "Text 2"],
    metadatas=[{"source": "1"}, {"source": "2"}],
    config=settings
)
```

### Async Operations

```python
# Async similarity search
results = await vector_store.asimilarity_search("query", k=5)

# Async get by IDs
docs = await vector_store.aget_by_ids(["id1", "id2"])
```

### Context Manager

```python
# Automatically closes connections when done
with Firebolt(config=settings) as vector_store:
    results = vector_store.similarity_search("query", k=5)
    # Connections are automatically closed
```

## API Reference

### Main Methods

#### `add_documents(documents, ids=None, batch_size=None, **kwargs)`
Add documents to the vector store.

#### `add_texts(texts, metadatas=None, ids=None, batch_size=None, **kwargs)`
Add texts to the vector store.

#### `similarity_search(query, k=4, filter=None, use_index=True, metadata_filter_k_multiplier=10, **kwargs)`
Search for similar documents by query text. The `use_index` parameter controls whether to use the vector search index (`True`, default) or brute force table scan (`False`). The `metadata_filter_k_multiplier` increases the number of candidates fetched from the index when filtering (default: 10).

#### `similarity_search_with_score(query, k=4, filter=None, use_index=True, metadata_filter_k_multiplier=10, **kwargs)`
Search for similar documents with similarity scores. The `use_index` parameter controls index vs brute force search (defaults to `True`). The `metadata_filter_k_multiplier` increases the number of candidates fetched from the index when filtering (default: 10).

#### `similarity_search_by_vector(embedding, k=4, filter=None, use_index=True, metadata_filter_k_multiplier=10, **kwargs)`
Search for similar documents using a vector embedding. The `use_index` parameter controls index vs brute force search (defaults to `True`). The `metadata_filter_k_multiplier` increases the number of candidates fetched from the index when filtering (default: 10).

#### `similarity_search_with_score_by_vector(embedding, k=4, filter=None, use_index=True, metadata_filter_k_multiplier=10, **kwargs)`
Search for similar documents by vector with similarity scores. The `use_index` parameter controls index vs brute force search (defaults to `True`). The `metadata_filter_k_multiplier` increases the number of candidates fetched from the index when filtering (default: 10).

#### `get_by_ids(ids)`
Retrieve documents by their IDs.

#### `delete(ids=None, filter=None, delete_all=False)`
Delete documents from the vector store.

#### `drop(drop_table=False)`
Drop the table and index (destructive operation).

#### `as_retriever(**kwargs)`
Create a retriever from the vector store.

#### `close()`
Close database connections.

## Environment Variables

You can configure the vector store using environment variables instead of passing `FireboltSettings`:

```bash
# Required
FIREBOLT_CLIENT_ID=your_client_id
FIREBOLT_CLIENT_SECRET=your_client_secret
FIREBOLT_ENGINE=your_engine
FIREBOLT_DB=your_database
FIREBOLT_ACCOUNT=your_account
FIREBOLT_TABLENAME=documents

# Optional
FIREBOLT_INDEX=documents_index
FIREBOLT_LLM_LOCATION=llm_api
FIREBOLT_BATCH_SIZE=32
FIREBOLT_API_ENDPOINT=https://api.firebolt.io
```

Then create the vector store without explicit config:

```python
vector_store = Firebolt()  # Uses environment variables
```

## Best Practices

### 1. Connection Management

The vector store uses two connections:
- **Read connection**: For search operations (autocommit enabled)
- **Write connection**: For write operations (autocommit disabled, uses transactions)

Always close connections when done:

```python
vector_store = Firebolt(config=settings)
try:
    # Use vector store
    results = vector_store.similarity_search("query")
finally:
    vector_store.close()
```

Or use the context manager:

```python
with Firebolt(config=settings) as vector_store:
    results = vector_store.similarity_search("query")
```

### 2. Batch Operations

For large datasets, use batch operations:

```python
# Process in batches
vector_store.add_documents(documents, batch_size=64)
```

### 3. Metadata Design

Design your metadata columns carefully:

```python
# Good: Specific, filterable columns
column_map = {
    "id": "id",
    "document": "document",
    "embedding": "embedding",
    "metadata": ["file_name", "page_number", "source", "author", "date"]
}
```

### 4. Index Selection

Choose the right metric for your use case:
- **Cosine similarity** (`vector_cosine_ops`): Best for normalized embeddings, most common
- **Inner product** (`vector_ip_ops`): Good for unnormalized embeddings
- **L2 squared distance** (`vector_l2sq_ops`): Good for distance-based applications

### 5. SQL Embeddings vs Client-Side Embeddings

**SQL Embeddings (Recommended, default):**
- Embeddings computed in Firebolt using `AI_EMBED_TEXT`
- No need to manage embeddings client-side
- Consistent with search-time embeddings
- **Requires LOCATION object setup** (see Prerequisites section)
- Set `use_sql_embeddings=True` (default) and provide `llm_location` parameter

**Client-Side Embeddings:**
- Embeddings computed using a LangChain embeddings model (e.g., OpenAI, HuggingFace)
- More control over embedding model
- Useful for testing or when LOCATION object is not available
- **LOCATION object not required**
- Set `use_sql_embeddings=False` and provide `embeddings` parameter

### 6. Error Handling

```python
try:
    vector_store.add_documents(documents)
except Exception as e:
    print(f"Error adding documents: {e}")
    # Connection will be rolled back automatically
```

### 7. Performance Optimization

- Use appropriate `batch_size` for your workload
- Create indexes on metadata columns used for filtering
- Use connection pooling for high-throughput applications
- Consider using external batch tools for initial data loading

## Troubleshooting

### Common Issues

**1. "No vector search index found"**
- Ensure you've created a vector search index on your table
- Or explicitly provide the `index` parameter in `FireboltSettings`

**2. "llm_location must be provided"**
- Create a LOCATION object in Firebolt
- Provide the `llm_location` parameter matching the LOCATION name

**3. "Authorization failed"**
- Verify your client ID and secret are correct
- Check that your credentials have access to the engine and database

## Additional Resources

- [Firebolt Documentation](https://docs.firebolt.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [Firebolt SDK](https://github.com/firebolt-db/firebolt-sdk-python)

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
