#!/bin/bash
# Build script for langchain-firebolt package

set -e

echo "Building langchain-firebolt package..."

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/

# Build the package
echo "Building wheel and source distribution..."
python3 -m build

echo "Build complete! Distribution files are in dist/"

