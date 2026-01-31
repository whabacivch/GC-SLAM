# Code Graph RAG MCP — Install from GitHub Only

**Use only the latest version from [GitHub Releases](https://github.com/er77/code-graph-rag-mcp/releases). Do not use the npm registry.**

## Why our earlier errors happened

- **"Unexpected token … is not valid JSON"** — Cursor/Codex expect **stdio stdout = JSON-RPC only**. Older versions wrote log lines (e.g. `[Config]`, `[CONDUCTOR]`, `Resource M...`) to **stdout**, so the client tried to parse them as JSON and failed. **As of v2.7.12**, stdout logs are redirected to **stderr** during MCP runs, so this is fixed when using the GitHub release.
- **"Segmentation fault (core dumped)"** — Can be (1) **better-sqlite3** native module mismatch (wrong Node ABI), or (2) heavy initialization before the MCP handshake in older versions. v2.7.12 defers heavy init until after handshake / first tool call. If you still see a segfault after installing 2.7.12+, run `npm rebuild better-sqlite3` in the global install dir (see [Troubleshooting](#troubleshooting)).

## 1. Install globally from a release `.tgz`

### Option A: Script (recommended)

From the project root:

```bash
bash tools/install_code_graph_rag_mcp.sh
```

This downloads the latest release `.tgz` from GitHub and runs `npm install -g` for you.

### Option B: Manual

1. Open [Releases](https://github.com/er77/code-graph-rag-mcp/releases) and download the latest `er77-code-graph-rag-mcp-*.tgz`.
2. Install globally:

   ```bash
   npm install -g ./er77-code-graph-rag-mcp-2.7.12.tgz
   ```

   (Use the exact filename you downloaded.)

3. Verify:

   ```bash
   code-graph-rag-mcp --version
   ```

## 2. Cursor / Codex integration

Use the **global binary** `code-graph-rag-mcp`, not `npx @er77/code-graph-rag-mcp`.

**Recommended (Codex/VSCode strict stdio):** Omit the directory argument and let the server learn the workspace root via `roots/list`:

| Field   | Value |
|---------|--------|
| Command | `code-graph-rag-mcp` |
| Args    | `[]` |

**Alternative (explicit path):** If your client does not provide roots, pass the project path:

| Field   | Value |
|---------|--------|
| Command | `code-graph-rag-mcp` |
| Args    | `["/home/will/Documents/Coding/Phantom Fellowship MIT/Impact_Project_v1"]` |

- Open **Cursor Settings → MCP** (or edit `~/.cursor/mcp.json`).
- Find the **code-graph-rag** (or **user-code-graph**) server and set the above.

**Example `~/.cursor/mcp.json` (recommended — no args):**

```json
{
  "mcpServers": {
    "code-graph-rag": {
      "command": "code-graph-rag-mcp",
      "args": [],
      "env": {
        "MCP_TIMEOUT": "80000"
      }
    }
  }
}
```

Restart Cursor (or reload MCP) after saving.

## 3. Other clients (reference)

- **Codex (recommended)**: `args = []`, workspace via roots/list — e.g. in config: `command = "code-graph-rag-mcp"`, `args = []`.
- **Claude Desktop**: use Inspector or add with `command`: `code-graph-rag-mcp`, `args`: `["/path/to/your/codebase"]` or `[]` if using roots.
- **Gemini**: `gemini mcp add-json code-graph-rag '{"command":"code-graph-rag-mcp","args":["/path/to/codebase"]}'`.

## 4. Requirements

- Node.js ≥ 24.0.0 (check with `node -v`). If you use nvm, run `nvm install 24` and `nvm use 24` before installing.
- If `code-graph-rag-mcp --version` fails with `ERR_MODULE_NOT_FOUND` for `onnxruntime-node`, install it globally: `npm install -g onnxruntime-node` (with Node 24 active).

## 5. Troubleshooting

| Issue | What to do |
|-------|------------|
| **Segfault after installing 2.7.12+** | Native module mismatch: run `npm rebuild better-sqlite3` in the install directory (global: e.g. `/usr/lib/node_modules/@er77/code-graph-rag-mcp` or `$(npm root -g)/@er77/code-graph-rag-mcp`). |
| **index / clean_index time out, transport closes** | Use **batch_index** with a small `maxFilesPerBatch` and keep calling with the returned `sessionId` until `done: true`. |
| **batch_index fails with agent_busy / memory_limit** | Raise coordinator/conductor limits: set `COORDINATOR_MEMORY_LIMIT`, `CONDUCTOR_MEMORY_LIMIT`, `COORDINATOR_MAX_MEMORY_MB`, `CONDUCTOR_MAX_MEMORY_MB` in env or in the server's `config/default.yaml`. For Node OOM, start with a larger heap: `NODE_OPTIONS="--max-old-space-size=4096" code-graph-rag-mcp`. |
| **Startup still fails** | Check the global tmp log: `/tmp/code-graph-rag-mcp/mcp-server-YYYY-MM-DD.log` (Linux/macOS). |
| **Logs on stdout for local debugging only** | Set `MCP_STDIO_ALLOW_STDOUT_LOGS=1` in env (not recommended for strict clients like Codex). |

## 6. Database and .gitignore

The server stores its SQLite DB under `./.code-graph-rag/vectors.db` (per repo). Add `/.code-graph-rag/` to your project's `.gitignore` (already added in this repo).

## 7. Same MCPs across Cursor, Claude Code, and Codex

All three clients are configured to use the same two MCPs so you get the same tools and shared memory:

| Client       | Config location | Servers |
|-------------|------------------|---------|
| **Cursor**  | `~/.cursor/mcp.json` | code-graph (Node 24 binary, `args: []`), memory-keeper (npx) |
| **Claude Code** | `~/.claude.json` → `projects["<project path>"].mcpServers` | Same two servers, same commands |
| **Codex**   | `~/.codex/config.toml` → `[mcp_servers.*]` | Same two servers, same commands |

**Shared memory:** mcp-memory-keeper uses `~/mcp-data/memory-keeper/` by default. As long as all three run the same memory-keeper (no custom storage path), they share the same memory store.
