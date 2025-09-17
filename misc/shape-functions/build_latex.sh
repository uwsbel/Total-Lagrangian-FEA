#!/bin/bash
set -e

# Run from this script's directory
cd "$(dirname "$0")"

# Run all Python files to generate shape function outputs
echo "Running all Python files..."
python3 beam_22.py
python3 beam_24.py
python3 beam_34.py
python3 shell_44.py
python3 shell_92.py
python3 hex_27.py
python3 tet_10.py
echo "Python files completed!"

# Build LaTeX document (run twice for refs)
echo "Building LaTeX document..."
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
echo "LaTeX build completed!"

