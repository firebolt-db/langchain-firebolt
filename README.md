# LangChain Firebolt Vector Store

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI version](https://badge.fury.io/py/langchain-firebolt.svg)](https://badge.fury.io/py/langchain-firebolt)

A LangChain vector store integration for Firebolt, enabling efficient similarity search and document management using Firebolt's vector search capabilities.

**Version:** 0.1.0

This package provides a standalone LangChain integration for Firebolt. It is not part of `langchain-community` and should be installed separately as `langchain-firebolt`.

## Features

- 🔍 **Vector Similarity Search**: Fast similarity search using Firebolt's HNSW vector indexes
- 📝 **SQL-Based Embeddings**: Generate embeddings directly in Firebolt using `AI_EMBED_TEXT`
- 🔄 **Client-Side Embeddings**: Support for precomputed embeddings from any LangChain-compatible embedding model
- 🎯 **Metadata Filtering**: Filter search results by metadata columns
- ⚡ **Async Support**: Full async/await support for all operations
- 🔗 **Retriever Integration**: Seamless integration with LangChain retrievers and chains
- 🛡️ **Connection Management**: Automatic connection handling with context manager support

## Table of Contents

- [Installation](#installation)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Environment Variables](#environment-variables)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Installation

```bash
pip install langchain-firebolt
```

**Note:** The `firebolt-sdk` dependency is automatically installed with `langchain-firebolt`. You don't need to install it separately.

## Prerequisites

### 1. Firebolt Account Setup

You need:
- A Firebolt account ([Sign up for a free trial](https://go.firebolt.io/signup))
- An engine running in your account
- A database created in your account
- Client credentials (client ID and secret)

### 2. Create LOCATION Object for LLM API

The Firebolt vector store uses Firebolt's `AI_EMBED_TEXT` SQL function to generate embeddings. This requires a LOCATION object to be created in Firebolt that points to your LLM service (e.g., Amazon Bedrock).

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

**Automatic Creation (Recommended):**

The Firebolt vector store can automatically create the table and vector index for you when they don't exist. When you initialize a `Firebolt` instance, it will:

1. **Check if the table exists** - If not, it creates the table using the structure defined in your `FireboltSettings.column_map`
2. **Check if the index exists** - If not, it creates the vector index using the `metric` and `embedding_dimension` from your `FireboltSettings`

The table structure is determined by your `column_map` configuration:
- `id` column: Used for document IDs (TEXT type, also used as PRIMARY INDEX)
- `document` column: Stores the document text (TEXT type)
- `embedding` column: Stores the vector embeddings (ARRAY(DOUBLE PRECISION NOT NULL) NOT NULL)
- `metadata` columns: Any columns listed in `column_map['metadata']` are added as TEXT columns (all metadata columns are created as TEXT type by default)

**Note:** If you need specific data types for metadata columns (e.g., INTEGER, DATE), you should create the table manually.

The index is created with:
- **Metric**: From `FireboltSettings.metric` (default: `vector_cosine_ops`)
- **Dimension**: From `FireboltSettings.embedding_dimension` (default: 256)
- **Index name**: From `FireboltSettings.index`, or auto-generated as `{table_name}_index` if not provided

**Example with automatic creation:**

```python
from langchain_firebolt import Firebolt, FireboltSettings

settings = FireboltSettings(
    id="your_client_id",
    secret="your_client_secret",
    engine_name="your_engine",
    database="my_database",
    account_name="your_account",
    table="documents",  # Table will be created if it doesn't exist
    index="documents_index",  # Optional: auto-generated as "documents_index" if not provided
    llm_location="llm_api",
    embedding_model="amazon.titan-embed-text-v2:0",
    embedding_dimension=256,  # Used for index creation
    metric="vector_cosine_ops",  # Used for index creation
    column_map={
        "id": "id",
        "document": "document",
        "embedding": "embedding",
        "metadata": ["file_name", "page_number", "source"]  # These columns will be created
    }
)

# Table and index are automatically created if they don't exist
vector_store = Firebolt(config=settings)
```

**Manual Creation (Optional):**

If you prefer to create the table and index manually, you can do so with the following SQL:

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

### Using Explicit Configuration

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
    llm_location="llm_api",  # Name of your LOCATION object
    embedding_model="amazon.titan-embed-text-v2:0",
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

### Using Environment Variables

Set your environment variables (or use a `.env` file):

```bash
export FIREBOLT_CLIENT_ID="your_client_id"
export FIREBOLT_CLIENT_SECRET="your_client_secret"
export FIREBOLT_ENGINE="your_engine"
export FIREBOLT_DB="my_database"
export FIREBOLT_ACCOUNT="your_account"
export FIREBOLT_TABLENAME="documents"
export FIREBOLT_LLM_LOCATION="llm_api"
```

```python
from langchain_firebolt import Firebolt, FireboltSettings
from langchain_core.documents import Document

# Settings automatically read from environment variables
# Only provide parameters that don't have environment variable support
settings = FireboltSettings(
    embedding_model="amazon.titan-embed-text-v2:0",  # Must be explicit
    embedding_dimension=256,  # Must be explicit
    metric="vector_cosine_ops"  # Optional: defaults to vector_cosine_ops
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

The `FireboltSettings` class configures the connection and behavior of the vector store. Many settings can be provided via environment variables, making it easy to manage configuration across different environments.

#### Required Parameters

- `id` (str): Firebolt client ID (env: `FIREBOLT_CLIENT_ID`)
- `secret` (str): Firebolt client secret (env: `FIREBOLT_CLIENT_SECRET`)
- `engine_name` (str): Name of the Firebolt engine (env: `FIREBOLT_ENGINE`)
- `database` (str): Name of the database (env: `FIREBOLT_DB`)
- `account_name` (str): Firebolt account name (env: `FIREBOLT_ACCOUNT`)
- `table` (str): Name of the table containing vectors (env: `FIREBOLT_TABLENAME`)
- `embedding_model` (str): Embedding model identifier (e.g., "amazon.titan-embed-text-v2:0") - **Must be provided explicitly, no environment variable**

#### Optional Parameters

- `index` (str, optional): Vector index name. If not provided, will be auto-detected from the database. (env: `FIREBOLT_INDEX`)
- `llm_location` (str, optional): Name of the LOCATION object in Firebolt. Required when `use_sql_embeddings=True`. (env: `FIREBOLT_LLM_LOCATION`)
- `embedding_dimension` (int): Dimension of embeddings. Defaults to 256. - **Must be provided explicitly, no environment variable**
- `batch_size` (int): Batch size for MERGE operations. Defaults to 32. (env: `FIREBOLT_BATCH_SIZE`)
- `metric` (str): Similarity metric. Options: `"vector_cosine_ops"` (default), `"vector_ip_ops"`, `"vector_l2sq_ops"`. - **Must be provided explicitly, no environment variable**
- `api_endpoint` (str, optional): Custom API endpoint. Defaults to Firebolt's cloud API. - **Must be provided explicitly, no environment variable**
- `column_map` (dict): Mapping of LangChain semantics to table columns. Defaults to:
  ```python
  {
      "id": "id",
      "document": "document",
      "embedding": "embedding",
      "metadata": []  # List of metadata column names
  }
  ```
  - **Must be provided explicitly, no environment variable**

#### Using Environment Variables

You can set configuration values using environment variables. `FireboltSettings` automatically reads from:
1. Environment variables (with explicit names like `FIREBOLT_CLIENT_ID`)
2. `.env` file in your project root (if present)
3. Explicit parameters passed to the constructor (takes precedence)

**Example: Using environment variables**

```bash
# Set environment variables
export FIREBOLT_CLIENT_ID="your_client_id"
export FIREBOLT_CLIENT_SECRET="your_client_secret"
export FIREBOLT_ENGINE="your_engine"
export FIREBOLT_DB="my_database"
export FIREBOLT_ACCOUNT="your_account"
export FIREBOLT_TABLENAME="documents"
export FIREBOLT_INDEX="documents_index"
export FIREBOLT_LLM_LOCATION="llm_api"
export FIREBOLT_BATCH_SIZE="64"
```

```python
from langchain_firebolt import Firebolt, FireboltSettings

# Settings will automatically read from environment variables
# Only provide parameters that don't have environment variables
settings = FireboltSettings(
    embedding_model="amazon.titan-embed-text-v2:0",  # Must be explicit
    embedding_dimension=256,  # Must be explicit
    metric="vector_cosine_ops"  # Must be explicit
)

vector_store = Firebolt(config=settings)
```

**Example: Using a `.env` file**

Create a `.env` file in your project root:

```bash
# .env
FIREBOLT_CLIENT_ID=your_client_id
FIREBOLT_CLIENT_SECRET=your_client_secret
FIREBOLT_ENGINE=your_engine
FIREBOLT_DB=my_database
FIREBOLT_ACCOUNT=your_account
FIREBOLT_TABLENAME=documents
FIREBOLT_INDEX=documents_index
FIREBOLT_LLM_LOCATION=llm_api
FIREBOLT_BATCH_SIZE=64
```

```python
from langchain_firebolt import Firebolt, FireboltSettings

# Settings automatically load from .env file
settings = FireboltSettings(
    embedding_model="amazon.titan-embed-text-v2:0",
    embedding_dimension=256
)

vector_store = Firebolt(config=settings)
```

**Example: Mixing environment variables and explicit parameters**

```python
from langchain_firebolt import Firebolt, FireboltSettings

# Explicit parameters override environment variables
settings = FireboltSettings(
    # These will use environment variables if set
    # id, secret, engine_name, database, account_name, table, etc.
    
    # These must be explicit (no env var support)
    embedding_model="amazon.titan-embed-text-v2:0",
    embedding_dimension=256,
    metric="vector_cosine_ops",
    
    # Override environment variable
    batch_size=128  # Overrides FIREBOLT_BATCH_SIZE if set
)

vector_store = Firebolt(config=settings)
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

The `Firebolt` vector store can be used as a retriever in LangChain chains:

```python
# Create a retriever with custom search parameters
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5, "filter": {"source": "docs"}}
)

# Use in a chain
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

result = qa_chain.invoke({"query": "What is machine learning?"})
```

#### Using FireboltRetriever Directly

You can also use `FireboltRetriever` directly:

```python
from langchain_firebolt import FireboltRetriever

retriever = FireboltRetriever(
    vector_store=vector_store,
    search_kwargs={"k": 10, "filter": {"file_name": "important.pdf"}}
)

# Get relevant documents
docs = retriever.get_relevant_documents("query text")

# Get documents with scores
docs_with_scores = retriever.get_relevant_documents_with_score("query text")
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

The vector store supports async operations for better performance in async applications:

```python
import asyncio

# Async similarity search
results = await vector_store.asimilarity_search("query", k=5)

# Async similarity search with score
results_with_scores = await vector_store.asimilarity_search_with_score("query", k=5)

# Async get by IDs
docs = await vector_store.aget_by_ids(["id1", "id2"])

# Async add documents
await vector_store.aadd_documents(documents)

# Async add texts
await vector_store.aadd_texts(texts=["Text 1", "Text 2"], metadatas=[{}, {}])

# Async delete
await vector_store.adelete(ids=["id1", "id2"])

# Example: Using in an async function
async def search_documents():
    vector_store = Firebolt(config=settings)
    try:
        results = await vector_store.asimilarity_search("machine learning", k=10)
        return results
    finally:
        vector_store.close()

# Run async function
results = asyncio.run(search_documents())
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

#### `similarity_search(query, k=4, filter=None, **kwargs)`
Search for similar documents by query text.

#### `similarity_search_with_score(query, k=4, filter=None, **kwargs)`
Search for similar documents with similarity scores.

#### `similarity_search_by_vector(embedding, k=4, filter=None, **kwargs)`
Search for similar documents using a vector embedding.

#### `get_by_ids(ids)`
Retrieve documents by their IDs.

#### `delete(ids=None, filter=None, delete_all=False)`
Delete documents from the vector store.

#### `drop(drop_table=False)`
Drop the table and index (destructive operation).

#### `as_retriever(**kwargs)`
Create a retriever from the vector store.

#### `asimilarity_search(query, k=4, filter=None, **kwargs)`
Async version of `similarity_search`.

#### `asimilarity_search_with_score(query, k=4, filter=None, **kwargs)`
Async version of `similarity_search_with_score`.

#### `aadd_documents(documents, ids=None, batch_size=None, **kwargs)`
Async version of `add_documents`.

#### `aadd_texts(texts, metadatas=None, ids=None, batch_size=None, **kwargs)`
Async version of `add_texts`.

#### `aget_by_ids(ids)`
Async version of `get_by_ids`.

#### `adelete(ids=None, filter=None, delete_all=False)`
Async version of `delete`.

#### `close()`
Close database connections.

## Environment Variables

Many `FireboltSettings` parameters can be configured using environment variables, making it easy to manage configuration across different environments (development, staging, production). The settings are automatically loaded from:

1. **Environment variables** (with explicit names like `FIREBOLT_CLIENT_ID`)
2. **`.env` file** in your project root (if present)
3. **Explicit parameters** passed to the constructor (takes precedence over environment variables)

### Supported Environment Variables

The following parameters can be set via environment variables:

```bash
# Required connection parameters
FIREBOLT_CLIENT_ID=your_client_id
FIREBOLT_CLIENT_SECRET=your_client_secret
FIREBOLT_ENGINE=your_engine
FIREBOLT_DB=your_database
FIREBOLT_ACCOUNT=your_account
FIREBOLT_TABLENAME=documents

# Optional parameters
FIREBOLT_INDEX=documents_index
FIREBOLT_LLM_LOCATION=llm_api
FIREBOLT_BATCH_SIZE=32
```

### Parameters That Must Be Explicit

The following parameters **cannot** be set via environment variables and must be provided explicitly:

- `embedding_model` - Embedding model identifier (e.g., "amazon.titan-embed-text-v2:0")
- `embedding_dimension` - Dimension of embeddings (default: 256)
- `metric` - Similarity metric (default: "vector_cosine_ops")
- `api_endpoint` - Custom API endpoint
- `column_map` - Column mapping configuration

### Usage Examples

**Using environment variables:**

```bash
export FIREBOLT_CLIENT_ID="your_client_id"
export FIREBOLT_CLIENT_SECRET="your_client_secret"
export FIREBOLT_ENGINE="your_engine"
export FIREBOLT_DB="my_database"
export FIREBOLT_ACCOUNT="your_account"
export FIREBOLT_TABLENAME="documents"
export FIREBOLT_LLM_LOCATION="llm_api"
```

```python
from langchain_firebolt import Firebolt, FireboltSettings

# Settings automatically read from environment variables
# Only provide parameters that don't have environment variable support
settings = FireboltSettings(
    embedding_model="amazon.titan-embed-text-v2:0",  # Must be explicit
    embedding_dimension=256,  # Must be explicit
    metric="vector_cosine_ops"  # Optional: defaults to vector_cosine_ops
)

vector_store = Firebolt(config=settings)
```

**Using a `.env` file:**

Create a `.env` file in your project root:

```bash
# .env
FIREBOLT_CLIENT_ID=your_client_id
FIREBOLT_CLIENT_SECRET=your_client_secret
FIREBOLT_ENGINE=your_engine
FIREBOLT_DB=my_database
FIREBOLT_ACCOUNT=your_account
FIREBOLT_TABLENAME=documents
FIREBOLT_INDEX=documents_index
FIREBOLT_LLM_LOCATION=llm_api
FIREBOLT_BATCH_SIZE=64
```

```python
from langchain_firebolt import Firebolt, FireboltSettings

# Settings automatically load from .env file
settings = FireboltSettings(
    embedding_model="amazon.titan-embed-text-v2:0",
    embedding_dimension=256
)

vector_store = Firebolt(config=settings)
```

**Mixing environment variables and explicit parameters:**

```python
from langchain_firebolt import Firebolt, FireboltSettings

# Explicit parameters override environment variables
settings = FireboltSettings(
    # Connection settings will use environment variables if set
    # id, secret, engine_name, database, account_name, table, etc.
    
    # These must be explicit (no env var support)
    embedding_model="amazon.titan-embed-text-v2:0",
    embedding_dimension=256,
    
    # Override environment variable
    batch_size=128  # Overrides FIREBOLT_BATCH_SIZE if set
)

vector_store = Firebolt(config=settings)
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

**SQL Embeddings (Recommended):**
- Embeddings computed in Firebolt using `AI_EMBED_TEXT`
- No need to manage embeddings client-side
- Consistent with search-time embeddings
- Requires LOCATION object setup

**Client-Side Embeddings:**
- More control over embedding model
- Useful for testing or when LOCATION object is not available
- Requires passing embeddings model to constructor

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
- **For large datasets**: Consider using external batch tools (e.g., Firebolt's bulk loading capabilities) for initial data loading to optimize tablet pruning and reduce the number of tablets
- The `add_documents()` and `add_texts()` methods are suitable for smaller datasets and incremental updates

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

**4. "Table does not exist"**
- Ensure the table is created before using the vector store
- Verify the `table` parameter matches your table name

**5. "Embedding dimension mismatch"**
- Ensure the `embedding_dimension` in `FireboltSettings` matches the dimension used when creating the index
- Verify the embedding model produces embeddings of the expected dimension

**6. "Connection timeout"**
- Check that your Firebolt engine is running
- Verify network connectivity to Firebolt
- Consider increasing connection timeout settings

## Additional Resources

- [Firebolt Documentation](https://docs.firebolt.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [Firebolt SDK](https://github.com/firebolt-db/firebolt-sdk-python)
- [GitHub Repository](https://github.com/firebolt-db/langchain-firebolt)
- [Issue Tracker](https://github.com/firebolt-db/langchain-firebolt/issues)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2024 Firebolt Analytics
