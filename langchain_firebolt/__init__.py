"""LangChain Firebolt Vector Store Integration."""

from langchain_firebolt.firebolt import (
    DEFAULT_MERGE_BATCH_SIZE,
    Firebolt,
    FireboltRetriever,
    FireboltSettings,
)

__all__ = [
    "DEFAULT_MERGE_BATCH_SIZE",
    "Firebolt",
    "FireboltRetriever",
    "FireboltSettings",
]

__version__ = "0.1.2"

