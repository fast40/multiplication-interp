#!/bin/sh
# Render paper/paper.tex -> paper.pdf using the local TeX Live docker image (docker/Dockerfile.latex).
set -e
cd "$(dirname "$0")"
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD":/work -w /work/paper mult-latex latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex >/dev/null
cp paper/paper.pdf paper.pdf
echo "rendered paper.pdf"
