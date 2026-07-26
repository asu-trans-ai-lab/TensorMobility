#!/usr/bin/env bash
# Build the TRB manuscript (requires pdflatex + biber).
set -e
pdflatex -interaction=nonstopmode main.tex
biber main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
