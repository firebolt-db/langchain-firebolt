"""Shared fixtures for Firebolt integration tests."""

import os
import pytest
from typing import Generator
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
# This allows integration tests to use .env file for configuration
load_dotenv()

# Import Firebolt SDK components
try:
    from firebolt.client.auth import ClientCredentials
    from firebolt.db import connect
except ImportError:
    raise ImportError(
        "Could not import firebolt-sdk python package. "
        "Please install it with `pip install firebolt-sdk`."
    )


@pytest.fixture(scope="function")
def firebolt_table_setup() -> Generator[dict, None, None]:
    """Set up LOCATION object for integration tests.
    
    This fixture:
    1. Creates the LOCATION object if it doesn't exist (if AWS credentials provided)
    2. Cleans up any existing table/index from previous test runs
    
    Note: Table and index creation is now handled automatically by the Firebolt class
    during initialization, so this fixture no longer creates them.
    
    Yields:
        dict: Configuration dictionary with table_name and index_name
    """
    # Get configuration from environment variables
    client_id = os.getenv("FIREBOLT_CLIENT_ID")
    client_secret = os.getenv("FIREBOLT_CLIENT_SECRET")
    engine_name = os.getenv("FIREBOLT_ENGINE")
    database = os.getenv("FIREBOLT_DB")
    account_name = os.getenv("FIREBOLT_ACCOUNT")
    api_endpoint = os.getenv("FIREBOLT_API_ENDPOINT")  # Optional custom API endpoint
    table = os.getenv("FIREBOLT_TABLENAME", "test_table")
    index = os.getenv("FIREBOLT_INDEX", f"{table}_index")
    embedding_dimension = int(os.getenv("FIREBOLT_EMBEDDING_DIMENSION", "256"))
    llm_location_name = os.getenv("FIREBOLT_LLM_LOCATION", "llm_api")
    metric = os.getenv("FIREBOLT_METRIC", "vector_cosine_ops")
    
    # AWS credentials for LOCATION object (optional - only needed if creating LOCATION)
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_session_token = os.getenv("AWS_SESSION_TOKEN")  # Optional, for temporary credentials
    
    # Skip if required environment variables are not set
    if not all([client_id, client_secret, engine_name, database, account_name]):
        pytest.skip("Firebolt credentials not provided in environment variables")
    
    # Connect to Firebolt using the same method as the main Firebolt class
    auth = ClientCredentials(
        client_id=client_id,
        client_secret=client_secret
    )
    connection_params = {
        "engine_name": engine_name,
        "database": database,
        "account_name": account_name,
        "auth": auth
    }
    # Add api_endpoint if specified (for custom domains/environments)
    # If api_endpoint contains "staging", use the staging API endpoint
    if api_endpoint:
        if "staging" in api_endpoint.lower():
            connection_params["api_endpoint"] = "https://api.staging.firebolt.io"
        else:
            connection_params["api_endpoint"] = api_endpoint
    connection = connect(**connection_params)
    
    cursor = connection.cursor()
    
    try:
        # Set database context
        cursor.execute(f"USE DATABASE {database}")
        
        # Always recreate LOCATION object for LLM API if AWS credentials are provided
        # This ensures we have fresh credentials and avoids stale credential issues
        if aws_access_key_id and aws_secret_access_key:
            # Check if LOCATION exists and drop it first to ensure fresh credentials
            cursor.execute(f"""
                SELECT COUNT(*) 
                FROM information_schema.locations 
                WHERE location_name = '{llm_location_name}'
            """)
            location_exists = cursor.fetchone()[0] > 0
            
            if location_exists:
                # Drop existing LOCATION to recreate with fresh credentials
                try:
                    cursor.execute(f"DROP LOCATION IF EXISTS {llm_location_name}")
                    print(f"Dropped existing LOCATION: {llm_location_name}")
                except Exception as e:
                    print(f"Warning: Could not drop LOCATION {llm_location_name}: {e}")
                    # Continue anyway - we'll try to create it
            
            # Build CREATE LOCATION statement with fresh credentials
            if aws_session_token:
                # Include session token if provided (for temporary credentials)
                create_location_sql = f"""CREATE LOCATION {llm_location_name} WITH
  SOURCE = AMAZON_BEDROCK
  CREDENTIALS = 
    (
    AWS_ACCESS_KEY_ID='{aws_access_key_id}'
    AWS_SECRET_ACCESS_KEY='{aws_secret_access_key}'
    AWS_SESSION_TOKEN='{aws_session_token}'
    )"""
            else:
                # Standard credentials without session token
                create_location_sql = f"""CREATE LOCATION {llm_location_name} WITH
  SOURCE = AMAZON_BEDROCK
  CREDENTIALS = 
    (
    AWS_ACCESS_KEY_ID='{aws_access_key_id}'
    AWS_SECRET_ACCESS_KEY='{aws_secret_access_key}'
    )"""
            try:
                cursor.execute(create_location_sql)
                print(f"Created LOCATION object: {llm_location_name} with fresh credentials")
            except Exception as e:
                error_msg = str(e).lower()
                # If location already exists (race condition), that's okay
                if "already exists" in error_msg or "duplicate" in error_msg:
                    print(f"LOCATION {llm_location_name} already exists (may have been created concurrently)")
                else:
                    print(f"Warning: Could not create LOCATION {llm_location_name}: {e}")
                    # Don't fail the test if LOCATION creation fails
        else:
            print(f"Skipping LOCATION creation: AWS credentials not provided")
        
        # Note: Table and index creation is now handled automatically by the Firebolt class
        # during initialization. The fixture only needs to ensure the LOCATION object exists.
        # Clean up any existing table/index from previous test runs (optional cleanup)
        print(f"Cleaning up any existing table/index {table} ...")
        try:
            cursor.execute(f"DROP INDEX IF EXISTS {index}")
            print(f"Dropped index: {index}")
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
            print(f"Dropped table: {table}")
        except Exception as e:
            print(f"Warning: Could not drop table/index: {e}")
        
        # Yield configuration
        yield {
            "table_name": table,
            "index_name": index,
            "embedding_dimension": embedding_dimension,
            "metric": metric,
        }
        
    finally:
        cursor.close()
        connection.close()

