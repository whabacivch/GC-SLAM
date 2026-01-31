#!/usr/bin/env bash
# Install code-graph-rag-mcp from GitHub Releases only (no npm registry).
# Usage: bash tools/install_code_graph_rag_mcp.sh

set -e

REPO="er77/code-graph-rag-mcp"
API_URL="https://api.github.com/repos/${REPO}/releases/latest"

echo "Fetching latest release from GitHub..."
RESP=$(curl -sL "$API_URL")
TAG=$(echo "$RESP" | grep -o '"tag_name": *"[^"]*"' | head -1 | sed 's/.*: *"\(.*\)".*/\1/')
TGZ_URL=$(echo "$RESP" | grep -o '"browser_download_url": *"[^"]*\.tgz"' | head -1 | sed 's/.*: *"\(.*\)".*/\1/')

if [ -z "$TGZ_URL" ]; then
  echo "Could not find .tgz asset in latest release. Check ${API_URL}"
  exit 1
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

TGZ_FILE="$TMPDIR/$(basename "$TGZ_URL")"
echo "Downloading $TGZ_URL ..."
curl -sL -o "$TGZ_FILE" "$TGZ_URL"

echo "Installing globally (npm install -g)..."
npm install -g "$TGZ_FILE"

echo ""
echo "Verifying..."
if ! code-graph-rag-mcp --version 2>/dev/null; then
  echo "First run failed (e.g. missing onnxruntime-node). Installing onnxruntime-node..."
  npm install -g onnxruntime-node
  code-graph-rag-mcp --version
fi
echo "Done. Use 'code-graph-rag-mcp' in Cursor MCP config (see tools/README_MCP.md)."
