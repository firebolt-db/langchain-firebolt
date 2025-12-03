# Packaging langchain-firebolt for PyPI

This document describes how to build and publish the `langchain-firebolt` package to PyPI.

## Package Structure

```
langchain_firebolt/
├── __init__.py
└── firebolt.py
```

## Building the Package

### Prerequisites

Install build tools:

```bash
pip install build twine
```

### Build Distribution

```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Build wheel and source distribution
python -m build
```

Or use the provided script:

```bash
chmod +x build_package.sh
./build_package.sh
```

This will create:
- `dist/langchain_firebolt-0.1.0-py3-none-any.whl` (wheel)
- `dist/langchain_firebolt-0.1.0.tar.gz` (source distribution)

## Testing the Package Locally

### Install in Development Mode

```bash
pip install -e .
```

### Test Installation

```bash
# Create a virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install the package
pip install dist/langchain_firebolt-0.1.0-py3-none-any.whl

# Test import
python -c "from langchain_firebolt import Firebolt, FireboltSettings; print('Success!')"
```

## Publishing to PyPI

### Test PyPI (Recommended First)

1. Create an account on [Test PyPI](https://test.pypi.org/)

2. Upload to Test PyPI:

```bash
twine upload --repository testpypi dist/*
```

3. Test installation from Test PyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ langchain-firebolt
```

### Production PyPI

1. Create an account on [PyPI](https://pypi.org/)

2. Upload to PyPI:

```bash
twine upload dist/*
```

You will be prompted for your PyPI credentials.

### Using API Tokens

For automation, you can use API tokens:

```bash
# Set environment variables
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-<your-token>

# Upload
twine upload dist/*
```

## Version Management

Update the version in:
- `pyproject.toml` (in `[project]` section)
- `setup.py` (in `version` parameter)
- `langchain_firebolt/__init__.py` (in `__version__`)

Follow [Semantic Versioning](https://semver.org/):
- MAJOR.MINOR.PATCH (e.g., 0.1.0)
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)

## Package Metadata

The package metadata is defined in `pyproject.toml`:

- **Name**: `langchain-firebolt`
- **Version**: `0.1.0`
- **Description**: LangChain integration for Firebolt vector store
- **Dependencies**: See `pyproject.toml` `[project.dependencies]`
- **Python Version**: >=3.8

## Installation

Once published, users can install the package with:

```bash
pip install langchain-firebolt
```

## Usage

After installation:

```python
from langchain_firebolt import Firebolt, FireboltSettings

settings = FireboltSettings(
    id="your_client_id",
    secret="your_client_secret",
    engine_name="your_engine",
    database="my_database",
    account_name="your_account",
    table="documents",
    llm_location="llm_api",
    embedding_model="amazon.titan-embed-text-v2:0",
)

vector_store = Firebolt(config=settings)
```

