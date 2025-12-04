"""Setup script for langchain-firebolt package."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="langchain-firebolt",
    version="0.1.0",
    author="Firebolt Analytics",
    author_email="support@firebolt.io",
    description="LangChain integration for Firebolt vector store",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/firebolt-db/langchain-firebolt",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "langchain-core>=0.1.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "firebolt-sdk>=1.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "langchain-tests>=0.0.1",
            "python-dotenv>=1.0.0",
        ],
    },
    keywords=["langchain", "firebolt", "vectorstore", "embeddings", "ai", "llm"],
)

