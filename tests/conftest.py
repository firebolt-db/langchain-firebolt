"""Shared pytest fixtures and utilities for Firebolt tests."""

import pytest
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List, Tuple, Optional
import uuid
import re


def _split_by_and_respecting_parentheses(where_clause: str) -> List[str]:
    """Split WHERE clause by AND, respecting nested parentheses.
    
    This properly handles cases like:
    - column1 = 'value1' AND column2 = 'value2'
    - column1 IN ('val1', 'val2') AND column2 = 'value2'
    - (column1 = 'value1') AND column2 = 'value2'
    
    Note: OR conditions are not supported by Firebolt's filter implementation,
    so we only split by AND at the top level.
    
    Args:
        where_clause: The WHERE clause string to split
        
    Returns:
        List of condition strings, split by AND at the top level only
    """
    conditions = []
    current_condition = []
    paren_depth = 0
    i = 0
    
    while i < len(where_clause):
        char = where_clause[i]
        
        # Track parentheses depth
        if char == '(':
            paren_depth += 1
            current_condition.append(char)
        elif char == ')':
            paren_depth -= 1
            current_condition.append(char)
        elif paren_depth == 0:
            # We're at top level, check for AND (with word boundaries)
            remaining = where_clause[i:]
            remaining_upper = remaining.upper()
            
            # Check for " AND " (with spaces) at word boundaries
            if remaining_upper.startswith(' AND '):
                # Found top-level AND, save current condition and start new one
                if current_condition:
                    conditions.append(''.join(current_condition).strip())
                    current_condition = []
                i += 5  # Skip " AND "
                continue
            else:
                current_condition.append(char)
        else:
            # Inside parentheses, just add the character
            current_condition.append(char)
        
        i += 1
    
    # Add the last condition
    if current_condition:
        conditions.append(''.join(current_condition).strip())
    
    return conditions


class CachedFakeEmbeddings:
    """Wrapper around FakeEmbeddings that caches embeddings by text for deterministic behavior."""
    
    def __init__(self, base_embeddings, text_to_embedding_cache: Dict[str, List[float]]):
        self.base_embeddings = base_embeddings
        self.cache = text_to_embedding_cache
    
    def embed_query(self, text: str) -> List[float]:
        """Get embedding for text, using cache if available."""
        if text in self.cache:
            return self.cache[text].copy()
        else:
            embedding = self.base_embeddings.embed_query(text)
            self.cache[text] = embedding.copy()
            return embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts, using cache when available."""
        embeddings = []
        for text in texts:
            if text in self.cache:
                embeddings.append(self.cache[text].copy())
            else:
                embedding = self.base_embeddings.embed_query(text)
                self.cache[text] = embedding.copy()
                embeddings.append(embedding)
        return embeddings
    
    def __getattr__(self, name):
        """Delegate all other attributes to the base embeddings model."""
        return getattr(self.base_embeddings, name)


def _parse_where_clause(sql: str) -> Dict[str, Any]:
    """Parse WHERE clause from SQL to extract filter conditions.
    
    Handles:
    - Equality: column = 'value'
    - IS NULL: column IS NULL
    - IN clauses: column IN ('val1', 'val2')
    - AND combinations (with proper nested parentheses handling)
    
    Returns a dictionary with column names as keys and filter values.
    """
    filter_dict = {}
    
    # Extract WHERE clause
    where_match = re.search(r'WHERE\s+(.+?)(?:\s+ORDER\s+BY|\s*$)', sql, re.IGNORECASE | re.DOTALL)
    if not where_match:
        return filter_dict
    
    where_clause = where_match.group(1).strip()
    
    # Split by AND, respecting nested parentheses
    conditions = _split_by_and_respecting_parentheses(where_clause)
    
    for condition in conditions:
        condition = condition.strip()
        
        # Skip if condition is wrapped in parentheses (might be a subquery or complex expression)
        # We'll try to parse it anyway, but strip outer parentheses if present
        if condition.startswith('(') and condition.endswith(')'):
            # Check if it's just parentheses around a simple condition
            inner = condition[1:-1].strip()
            # If inner doesn't contain AND/OR at top level, use it
            if ' AND ' not in inner.upper() and ' OR ' not in inner.upper():
                condition = inner
        
        # Handle IS NULL
        is_null_match = re.match(r'(\w+)\s+IS\s+NULL', condition, re.IGNORECASE)
        if is_null_match:
            column = is_null_match.group(1)
            filter_dict[column] = None
            continue
        
        # Handle IN clause - need to handle nested parentheses properly
        in_match = re.search(r'(\w+)\s+IN\s*\(', condition, re.IGNORECASE)
        if in_match:
            column = in_match.group(1)
            # Find the matching closing parenthesis
            start_pos = in_match.end() - 1  # Position of opening (
            paren_depth = 0
            end_pos = start_pos
            
            for i in range(start_pos, len(condition)):
                if condition[i] == '(':
                    paren_depth += 1
                elif condition[i] == ')':
                    paren_depth -= 1
                    if paren_depth == 0:
                        end_pos = i
                        break
            
            if end_pos > start_pos:
                values_str = condition[start_pos + 1:end_pos]
                # Parse values (handle quoted strings and numbers)
                values = []
                # Split by comma, but respect quoted strings
                current_value = []
                in_quotes = False
                quote_char = None
                
                for char in values_str:
                    if not in_quotes and (char == "'" or char == '"'):
                        in_quotes = True
                        quote_char = char
                        current_value.append(char)
                    elif in_quotes and char == quote_char:
                        in_quotes = False
                        quote_char = None
                        current_value.append(char)
                    elif not in_quotes and char == ',':
                        val = ''.join(current_value).strip().strip("'\"")
                        if val:
                            try:
                                if '.' in val:
                                    values.append(float(val))
                                else:
                                    values.append(int(val))
                            except ValueError:
                                values.append(val)
                        current_value = []
                    else:
                        current_value.append(char)
                
                # Add the last value
                if current_value:
                    val = ''.join(current_value).strip().strip("'\"")
                    if val:
                        try:
                            if '.' in val:
                                values.append(float(val))
                            else:
                                values.append(int(val))
                        except ValueError:
                            values.append(val)
                
                filter_dict[column] = values
                continue
        
        # Handle equality - extract value, handling quoted strings
        # Use a more robust regex that handles quoted strings
        eq_match = re.search(r'(\w+)\s*=\s*((?:\'[^\']*(?:\'\'[^\']*)*\'|"[^"]*(?:""[^"]*)*"|\d+(?:\.\d+)?|\w+))', condition, re.IGNORECASE)
        if eq_match:
            column = eq_match.group(1)
            value_str = eq_match.group(2).strip()
            
            # Handle quoted strings (single or double quotes)
            if (value_str.startswith("'") and value_str.endswith("'")):
                value = value_str[1:-1]  # Remove outer quotes
                # Handle escaped single quotes ('' becomes ')
                value = value.replace("''", "'")
            elif (value_str.startswith('"') and value_str.endswith('"')):
                value = value_str[1:-1]  # Remove outer quotes
                # Handle escaped double quotes ("" becomes ")
                value = value.replace('""', '"')
            else:
                # Try to convert to number if possible
                try:
                    if '.' in value_str:
                        value = float(value_str)
                    else:
                        value = int(value_str)
                except ValueError:
                    value = value_str
            
            filter_dict[column] = value
    
    return filter_dict


class MockFireboltDatabase:
    """In-memory mock database for Firebolt operations."""
    
    def __init__(self, embeddings_model=None):
        self.documents: Dict[str, Dict[str, Any]] = {}  # id -> {id, document, embedding, metadata_cols}
        self.semantic_indexes: Dict[str, str] = {}  # table_name -> index_name
        self.embeddings_model = embeddings_model  # Store embeddings model for query embedding
        self.embedding_to_text: Dict[tuple, str] = {}  # Store mapping of embedding tuples to text for query inference
        self.text_to_embedding: Dict[str, List[float]] = {}  # Cache embeddings by text for deterministic behavior
        
    def add_document(self, doc_id: str, document: str, embedding: List[float], metadata: Dict[str, Any] = None):
        """Add or update a document in the mock database (upsert behavior)."""
        metadata = metadata or {}
        
        # Cache embedding by text for deterministic behavior
        # This ensures the same text always produces the same embedding
        if document not in self.text_to_embedding:
            self.text_to_embedding[document] = embedding.copy()
        else:
            # Use cached embedding to ensure consistency
            embedding = self.text_to_embedding[document]
        
        # If document exists, update it; otherwise add new
        if doc_id in self.documents:
            # Update existing document
            existing = self.documents[doc_id]
            existing['document'] = document
            existing['embedding'] = embedding
            # Replace metadata entirely (new metadata takes full precedence)
            # This matches the behavior of updating a document
            existing['metadata'] = metadata.copy()
        else:
            # Add new document
            self.documents[doc_id] = {
                'id': doc_id,
                'document': document,
                'embedding': embedding,
                'metadata': metadata.copy()
            }
        
        # Store mapping of embedding to text for query inference
        # Use a rounded tuple to handle floating point precision
        # Store multiple keys: first 10, first 20, and full embedding (rounded) for better matching
        embedding_key_10 = tuple(round(x, 6) for x in embedding[:10])
        embedding_key_20 = tuple(round(x, 6) for x in embedding[:20])
        embedding_key_full = tuple(round(x, 6) for x in embedding[:50])  # Use first 50 for better matching
        self.embedding_to_text[embedding_key_10] = document
        self.embedding_to_text[embedding_key_20] = document
        self.embedding_to_text[embedding_key_full] = document
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            return True
        return False
    
    def delete_by_filter(self, filter_dict: Dict[str, Any]) -> int:
        """Delete documents matching filter."""
        deleted = 0
        to_delete = []
        # Sort by doc_id first to ensure deterministic processing order
        for doc_id, doc in sorted(self.documents.items()):
            match = True
            for key, value in filter_dict.items():
                if key not in doc.get('metadata', {}) or doc['metadata'][key] != value:
                    match = False
                    break
            if match:
                to_delete.append(doc_id)
        
        for doc_id in to_delete:
            del self.documents[doc_id]
            deleted += 1
        
        return deleted
    
    def delete_all(self) -> int:
        """Delete all documents."""
        count = len(self.documents)
        self.documents.clear()
        return count
    
    def similarity_search(self, query_embedding: List[float], k: int, filter_dict: Dict[str, Any] = None, metadata_cols: List[str] = None, query_text: str = None) -> List[Tuple]:
        """Perform similarity search and return rows in format: [id, document, metadata_cols..., dist]"""
        # Simple cosine similarity calculation
        results = []
        metadata_cols = metadata_cols or []
        
        # If we have query text, use cached embedding for that text to ensure deterministic matching
        # This is critical because FakeEmbeddings is non-deterministic - same text produces different embeddings
        # By using cached embeddings, we ensure the same text always uses the same embedding
        if query_text and query_text in self.text_to_embedding:
            query_embedding = self.text_to_embedding[query_text]
        elif query_text and self.embeddings_model:
            # Cache the embedding for this query text
            query_embedding = self.get_embedding_for_text(query_text)
        
        # Sort by doc_id first to ensure deterministic processing order
        for doc_id, doc in sorted(self.documents.items()):
            # Apply filter if provided
            if filter_dict:
                match = True
                for key, value in filter_dict.items():
                    doc_value = doc.get('metadata', {}).get(key)
                    # Handle type coercion for comparison (e.g., '2' == 2)
                    if doc_value != value:
                        # Try type conversion for comparison
                        try:
                            if isinstance(value, int) and isinstance(doc_value, str):
                                if int(doc_value) != value:
                                    match = False
                                    break
                            elif isinstance(value, str) and isinstance(doc_value, int):
                                if doc_value != int(value):
                                    match = False
                                    break
                            else:
                                match = False
                                break
                        except (ValueError, TypeError):
                            match = False
                            break
                if not match:
                    continue
            
            # Calculate cosine similarity using vector math
            embedding = doc['embedding']
            doc_text = doc.get('document', '')
            
            # Check for exact embedding match first (before calculating distance)
            # This is critical for FakeEmbeddings where same text = same embedding
            # This works even when query_text is None
            distance = None
            if len(query_embedding) == len(embedding):
                # Check if embeddings are identical (exact match)
                # For FakeEmbeddings, same text should produce identical embeddings
                # So we can check for exact equality first, then use a small threshold
                if query_embedding == embedding:
                    distance = 0.0  # Exact match
                else:
                    max_diff = max(abs(a - b) for a, b in zip(query_embedding, embedding))
                    if max_diff < 1e-10:  # Very small threshold for floating point comparison
                        distance = 0.0  # Exact match gets distance 0.0
            
            # If not an exact match, calculate cosine distance
            if distance is None:
                # Ensure embeddings have the same length
                min_len = min(len(query_embedding), len(embedding))
                if min_len == 0:
                    distance = 1.0
                else:
                    query_emb = query_embedding[:min_len]
                    doc_emb = embedding[:min_len]
                    
                    # Calculate cosine similarity: dot product / (norm_a * norm_b)
                    dot_product = sum(a * b for a, b in zip(query_emb, doc_emb))
                    norm_a = sum(a * a for a in query_emb) ** 0.5
                    norm_b = sum(b * b for b in doc_emb) ** 0.5
                    
                    if (norm_a * norm_b) > 0:
                        similarity = dot_product / (norm_a * norm_b)
                        distance = 1 - similarity  # cosine distance
                    else:
                        distance = 1.0
                    
                    # If embeddings are very similar (distance < 0.0001), treat as exact match
                    if distance < 0.0001:
                        distance = 0.0  # Exact match gets distance 0
                
                # If we have query text, use text-based matching FIRST (before embedding-based)
                # This ensures that exact text matches always come first, regardless of embedding similarity
                if query_text:
                    query_text_clean = query_text.strip().lower()
                    doc_text_clean = doc_text.strip().lower()
                    
                    # Exact text match gets distance 0.0 (highest priority)
                    if query_text_clean == doc_text_clean:
                        distance = 0.0
                    # Query is exact substring of document (or vice versa) gets very low distance
                    elif query_text_clean in doc_text_clean or doc_text_clean in query_text_clean:
                        distance = min(distance, 0.001)  # Very low distance for substring matches
                    # If query text starts with document text or vice versa, boost it
                    elif query_text_clean.startswith(doc_text_clean) or doc_text_clean.startswith(query_text_clean):
                        distance = min(distance, 0.01)  # Low distance for prefix matches
                    # If documents share significant words, give a small boost
                    elif query_text_clean.split() and doc_text_clean.split():
                        query_words = set(query_text_clean.split())
                        doc_words = set(doc_text_clean.split())
                        common_words = query_words & doc_words
                        if common_words:
                            # Boost based on word overlap
                            overlap_ratio = len(common_words) / max(len(query_words), len(doc_words))
                            distance = min(distance, distance * (1 - overlap_ratio * 0.5))
                    # If no text match, ensure non-matching documents have higher distance
                    else:
                        distance = max(distance, 0.1)  # Ensure non-matches have at least 0.1 distance
                
            
            # Build row: [id, document, metadata_cols..., dist]
            # The id column will be extracted by Firebolt and put in metadata['id']
            # So we need to use the original metadata['id'] value to preserve type
            original_metadata = doc.get('metadata', {})
            
            # Use original metadata['id'] if it exists (preserves type like int), otherwise use doc_id
            if 'id' in original_metadata:
                row_id = original_metadata['id']  # Preserve original type (int, string, etc.)
            else:
                row_id = doc_id  # Fallback to string doc_id
            
            # Convert to string for the row (Firebolt will convert back based on metadata)
            # Actually, let's preserve the type - if it's int, keep it int
            row = [row_id, str(doc['document'])]
            # Add metadata columns in the order specified, preserving original types
            if metadata_cols:
                for col_name in metadata_cols:
                    row.append(original_metadata.get(col_name))
            row.append(distance)
            results.append((row, distance, original_metadata))
        
        # Sort by distance (primary) and then by document content (secondary) for deterministic ordering
        # This ensures that exact matches (distance 0.0) are ordered consistently
        # Also, if we have query text, prioritize documents whose text matches the query
        if query_text:
            query_text_clean = query_text.strip().lower()
            def sort_key(x):
                row, dist, metadata = x
                doc_text = row[1] if len(row) > 1 else ''
                doc_text_clean = doc_text.strip().lower()
                doc_id = row[0] if len(row) > 0 else ''
                # Exact match gets highest priority (distance 0.0, then text match)
                if doc_text_clean == query_text_clean:
                    return (0.0, 0, doc_text, doc_id)  # Distance 0, exact match flag, then text, then id
                # Substring match gets second priority
                elif query_text_clean in doc_text_clean or doc_text_clean in query_text_clean:
                    return (dist, 1, doc_text, doc_id)  # Original distance, substring match flag, then text, then id
                else:
                    return (dist, 2, doc_text, doc_id)  # Original distance, no match flag, then text, then id
            results.sort(key=sort_key)
        else:
            # Sort by distance (ascending - lower distance = more similar), then by document text, then by doc_id for deterministic ordering
            # x[0] = row = [id, document, metadata_cols..., distance]
            # x[1] = distance (the actual distance value we want to sort by)
            # x[0][0] = id, x[0][1] = document text
            results.sort(key=lambda x: (x[1]))
        return [row for row, _, _ in results[:k]]
    
    def get_semantic_index(self, table_name: str) -> str:
        """Get semantic index name for a table."""
        return self.semantic_indexes.get(table_name, f"{table_name}_index")
    
    def reset(self):
        """Reset the mock database to empty state."""
        self.documents.clear()
        self.semantic_indexes.clear()
        self.embedding_to_text.clear()
        self.text_to_embedding.clear()
    
    def get_embedding_for_text(self, text: str) -> List[float]:
        """Get embedding for text, using cache if available for deterministic behavior."""
        if text in self.text_to_embedding:
            return self.text_to_embedding[text].copy()
        elif self.embeddings_model:
            embedding = self.embeddings_model.embed_query(text)
            self.text_to_embedding[text] = embedding.copy()
            return embedding
        else:
            return [0.0] * 256


@pytest.fixture
def mock_firebolt_db(embedding_openai):
    """Create a mock Firebolt database instance."""
    db = MockFireboltDatabase(embeddings_model=embedding_openai)
    yield db
    # Reset after each test to ensure test isolation
    db.reset()


@pytest.fixture
def mock_firebolt_connection(mock_firebolt_db):
    """Create a mock Firebolt connection with database."""
    mock_connection = MagicMock()
    
    def create_cursor():
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        # Set up cursor.description for column names (for metadata extraction)
        mock_cursor.description = None
        
        def execute(sql: str):
            # Handle INSERT operations
            if sql.strip().upper().startswith("INSERT"):
                # Count rows being inserted (rough estimate)
                values_count = sql.upper().count("VALUES") + sql.upper().count("),")
                mock_cursor.rowcount = max(1, values_count)
            
            # Handle DELETE operations
            elif sql.strip().upper().startswith("DELETE"):
                if "DELETE FROM" in sql.upper():
                    if "WHERE" not in sql.upper() or "WHERE 1=1" in sql.upper():
                        # DELETE ALL
                        count = mock_firebolt_db.delete_all()
                        mock_cursor.rowcount = count
                    elif "IN (" in sql.upper():
                        # DELETE BY IDs
                        ids_match = re.search(r"IN\s*\(([^)]+)\)", sql, re.IGNORECASE)
                        if ids_match:
                            ids_str = ids_match.group(1)
                            # Parse quoted IDs
                            ids = [id.strip().strip("'\"") for id in ids_str.split(",")]
                            count = sum(1 for doc_id in ids if mock_firebolt_db.delete_document(doc_id.strip()))
                            mock_cursor.rowcount = count
                    else:
                        # DELETE BY FILTER - parse WHERE clause
                        filter_dict = _parse_where_clause(sql)
                        if filter_dict:
                            count = mock_firebolt_db.delete_by_filter(filter_dict)
                            mock_cursor.rowcount = count
                        else:
                            mock_cursor.rowcount = 0
            
            # Handle SELECT for semantic index
            elif "information_schema.indexes" in sql:
                table_match = re.search(r"table_name\s*=\s*['\"]([^'\"]+)['\"]", sql, re.IGNORECASE)
                if table_match:
                    table_name = table_match.group(1)
                    index_name = mock_firebolt_db.get_semantic_index(table_name)
                    mock_cursor.fetchall.return_value = [(index_name,)]
                else:
                    mock_cursor.fetchall.return_value = []
            
            # Handle SET and USE statements
            elif sql.strip().upper().startswith("SET") or sql.strip().upper().startswith("USE"):
                pass  # No-op for session settings
            
            # Handle similarity search queries (vector_search)
            elif "vector_search" in sql.lower() or "VECTOR_COSINE_DISTANCE" in sql.upper():
                # Extract embedding from SQL (simplified)
                embedding_match = re.search(r'\[([^\]]+)\]', sql)
                if embedding_match:
                    embedding_str = embedding_match.group(1)
                    try:
                        query_embedding = [float(x.strip()) for x in embedding_str.split(",")]
                    except:
                        query_embedding = [0.0] * 256
                else:
                    query_embedding = [0.0] * 256
                
                # Try to extract query text from SQL if AI_EMBED_TEXT is used
                # This helps us identify exact text matches
                query_text = None
                if mock_firebolt_db.embeddings_model:
                    # Check if AI_EMBED_TEXT is in the SQL (case-insensitive)
                    sql_upper = sql.upper()
                    if "AI_EMBED_TEXT" in sql_upper:
                        # Find the position of INPUT_TEXT in the SQL
                        input_text_pos = sql_upper.find("INPUT_TEXT")
                        if input_text_pos != -1:
                            # Find the arrow operator after INPUT_TEXT
                            arrow_pos = sql.find("=>", input_text_pos)
                            if arrow_pos != -1:
                                # Find the opening quote after the arrow
                                quote_start = sql.find("'", arrow_pos)
                                if quote_start != -1:
                                    # Find the closing quote (handle escaped quotes)
                                    quote_end = quote_start + 1
                                    while quote_end < len(sql):
                                        if sql[quote_end] == "'":
                                            # Check if it's escaped (two quotes in a row)
                                            if quote_end + 1 < len(sql) and sql[quote_end + 1] == "'":
                                                quote_end += 2  # Skip escaped quote
                                            else:
                                                # Found the closing quote
                                                break
                                        else:
                                            quote_end += 1
                                    
                                    # Extract the text between quotes
                                    if quote_end < len(sql):
                                        query_text = sql[quote_start + 1:quote_end]
                                        # Unescape single quotes (replace '' with ')
                                        query_text = query_text.replace("''", "'")
                    
                
                # If we couldn't extract query text from SQL, try to match the query embedding
                # against cached document embeddings to find the corresponding text
                # This works because we cache embeddings by text in add_document
                if not query_text and mock_firebolt_db.text_to_embedding:
                    # Try to find which cached text produces an embedding that matches the query embedding
                    # Check for exact matches first (for FakeEmbeddings with caching, same text = same embedding)
                    for text, cached_emb in mock_firebolt_db.text_to_embedding.items():
                        if len(cached_emb) == len(query_embedding):
                            # Check if embeddings are identical
                            if cached_emb == query_embedding:
                                query_text = text
                                break
                            # Check if they're very close (within floating point precision)
                            max_diff = max(abs(a - b) for a, b in zip(query_embedding, cached_emb))
                            if max_diff < 1e-10:
                                query_text = text
                                break
                    
                    # If no exact match, find the closest match by cosine similarity
                    if not query_text:
                        best_match_text = None
                        best_similarity = -1.0
                        for text, cached_emb in mock_firebolt_db.text_to_embedding.items():
                            if len(cached_emb) == len(query_embedding):
                                # Calculate cosine similarity
                                dot = sum(a * b for a, b in zip(query_embedding, cached_emb))
                                norm_q = sum(a * a for a in query_embedding) ** 0.5
                                norm_c = sum(b * b for b in cached_emb) ** 0.5
                                if (norm_q * norm_c) > 0:
                                    similarity = dot / (norm_q * norm_c)
                                    if similarity > best_similarity:
                                        best_similarity = similarity
                                        best_match_text = text
                        
                        # Only use the best match if it's very close (similarity > 0.99)
                        if best_match_text and best_similarity > 0.99:
                            query_text = best_match_text
                            # Use the cached embedding for this text to ensure exact matching
                            query_embedding = mock_firebolt_db.text_to_embedding[best_match_text]
                
                # Extract k value
                k_match = re.search(r'vector_search[^,]*,\s*\[[^\]]+\]\s*,\s*(\d+)', sql, re.IGNORECASE)
                k = int(k_match.group(1)) if k_match else 4
                
                # Extract filter if present - parse WHERE clause
                filter_dict = None
                if "WHERE" in sql.upper():
                    filter_dict = _parse_where_clause(sql)
                
                # Extract metadata columns from the SQL SELECT clause
                # The query structure is: SELECT id, document, [metadata_cols...], FUNC(...) AS dist FROM ...
                # We need to parse the columns between document and dist, but carefully
                # because the distance function contains brackets with embedding values
                metadata_cols = []
                
                select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
                if select_match:
                    select_clause = select_match.group(1)
                    
                    # Parse column names properly, respecting parentheses and brackets
                    # We need to split by comma only at the top level (not inside () or [])
                    columns = []
                    current_col = []
                    depth = 0  # Track nesting depth of () and []
                    
                    for char in select_clause:
                        if char in '([':
                            depth += 1
                            current_col.append(char)
                        elif char in ')]':
                            depth -= 1
                            current_col.append(char)
                        elif char == ',' and depth == 0:
                            # Top-level comma - end of column
                            col_str = ''.join(current_col).strip()
                            if col_str:
                                columns.append(col_str)
                            current_col = []
                        else:
                            current_col.append(char)
                    
                    # Don't forget the last column
                    col_str = ''.join(current_col).strip()
                    if col_str:
                        columns.append(col_str)
                    
                    # Extract just the column name (handle aliases like "col AS alias" and functions)
                    simple_columns = []
                    for col in columns:
                        # Check if it has an alias (e.g., "FUNC(...) AS dist")
                        as_match = re.search(r'\s+AS\s+(\w+)\s*$', col, re.IGNORECASE)
                        if as_match:
                            # Use the alias
                            simple_columns.append(as_match.group(1))
                        elif '(' in col:
                            # It's a function call without alias, skip or use a default name
                            simple_columns.append('_func_result')
                        else:
                            # Simple column name
                            simple_columns.append(col.strip())
                    
                    columns = simple_columns
                    
                    # Expected structure: [id_col, document_col, metadata_cols..., dist]
                    # Skip first 2 (id, document) and last 1 (dist) to get metadata columns
                    if len(columns) > 3:
                        # Columns between document and dist are metadata columns
                        metadata_cols = columns[2:-1]
                
                # Set up cursor.description with column names for metadata extraction
                # Format: (name, type_code, display_size, internal_size, precision, scale, null_ok)
                # id, document, metadata_cols..., dist
                description = [('id', None), ('document', None)]
                description.extend([(col, None) for col in metadata_cols])
                description.append(('dist', None))
                mock_cursor.description = description
                
                results = mock_firebolt_db.similarity_search(query_embedding, k, filter_dict, metadata_cols, query_text=query_text)
                mock_cursor.fetchall.return_value = results
            
            # Handle SELECT queries for get_by_ids (WHERE id IN (...))
            elif sql.strip().upper().startswith("SELECT") and "IN (" in sql.upper():
                # Extract table name
                from_match = re.search(r'FROM\s+(\S+)', sql, re.IGNORECASE)
                if from_match:
                    # Extract IDs from IN clause
                    in_match = re.search(r'IN\s*\(([^)]+)\)', sql, re.IGNORECASE)
                    if in_match:
                        ids_str = in_match.group(1)
                        # Parse quoted IDs
                        ids = [id.strip().strip("'\"") for id in ids_str.split(",")]
                        
                        # Extract column names from SELECT clause
                        select_match = re.search(r'SELECT\s+([^,]+(?:,\s*[^,]+)*)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
                        if select_match:
                            select_clause = select_match.group(1)
                            # Parse column names (simplified - assumes no functions)
                            columns = [col.strip() for col in select_clause.split(",")]
                            
                            # Set up cursor.description with column names
                            description = [(col, None) for col in columns]
                            mock_cursor.description = description
                            
                            # Get documents by IDs from mock database
                            results = []
                            for doc_id in ids:
                                doc_id_clean = doc_id.strip().strip("'\"")
                                if doc_id_clean in mock_firebolt_db.documents:
                                    doc = mock_firebolt_db.documents[doc_id_clean]
                                    # Build row: [id, document, metadata_cols...]
                                    row = [doc.get('id', doc_id_clean), doc.get('document', '')]
                                    # Add metadata columns
                                    metadata = doc.get('metadata', {})
                                    for col in columns[2:]:  # Skip id and document columns
                                        row.append(metadata.get(col))
                                    results.append(tuple(row))
                            
                            mock_cursor.fetchall.return_value = results
                        else:
                            mock_cursor.fetchall.return_value = []
                    else:
                        mock_cursor.fetchall.return_value = []
                else:
                    mock_cursor.fetchall.return_value = []
            
            # Handle other SELECT queries (e.g., for existing document metadata in MERGE)
            elif sql.strip().upper().startswith("SELECT"):
                # Extract table name and WHERE clause
                from_match = re.search(r'FROM\s+(\S+)', sql, re.IGNORECASE)
                if from_match and "WHERE" in sql.upper():
                    # Parse WHERE clause to get IDs
                    where_match = re.search(r'WHERE\s+(\S+)\s+IN\s*\(([^)]+)\)', sql, re.IGNORECASE)
                    if where_match:
                        ids_str = where_match.group(2)
                        ids = [id.strip().strip("'\"") for id in ids_str.split(",")]
                        
                        # Extract column names from SELECT clause
                        select_match = re.search(r'SELECT\s+([^,]+(?:,\s*[^,]+)*)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
                        if select_match:
                            select_clause = select_match.group(1)
                            columns = [col.strip() for col in select_clause.split(",")]
                            
                            # Set up cursor.description
                            description = [(col, None) for col in columns]
                            mock_cursor.description = description
                            
                            # Get documents by IDs
                            results = []
                            for doc_id in ids:
                                doc_id_clean = doc_id.strip().strip("'\"")
                                if doc_id_clean in mock_firebolt_db.documents:
                                    doc = mock_firebolt_db.documents[doc_id_clean]
                                    # Build row based on selected columns
                                    row = []
                                    for col in columns:
                                        if col == 'id':
                                            row.append(doc.get('id', doc_id_clean))
                                        elif col == 'document':
                                            row.append(doc.get('document', ''))
                                        else:
                                            # Metadata column
                                            metadata = doc.get('metadata', {})
                                            row.append(metadata.get(col))
                                    results.append(tuple(row))
                            
                            mock_cursor.fetchall.return_value = results
                        else:
                            mock_cursor.fetchall.return_value = []
                    else:
                        mock_cursor.fetchall.return_value = []
                else:
                    mock_cursor.fetchall.return_value = []
            else:
                mock_cursor.fetchall.return_value = []
        
        mock_cursor.execute = execute
        mock_cursor.fetchall.return_value = []
        mock_cursor.fetchone.return_value = None
        return mock_cursor
    
    mock_connection.cursor = create_cursor
    mock_connection.close = MagicMock()
    mock_connection._mock_db = mock_firebolt_db
    
    return mock_connection

