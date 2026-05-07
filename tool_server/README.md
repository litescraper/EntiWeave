## Tools Server (FastAPI)

This folder exposes a lightweight HTTP tools server used by the agent to perform web search and page fetching/parsing.

### Endpoints

All endpoints require an `X-API-Key` header for authentication.

- **GET `/search`**: Search the web using the configured searcher.
  - **Query params**: `q` (string, required), `num` (int, default 10), `debug` (0/1).
  - **Searchers**: `jina`, `ddg` (DuckDuckGo), `serper`, `local` — set at startup via `--searcher-type`.

- **GET `/click`**: Fetch and parse content from a URL.
  - **Query params**: `url` (string, required), `debug` (0/1).
  - **Retrievers**: `custom` (direct fetch with specialized parsers for HTML, PDF, JSON, Wikipedia), `jina` (r.jina.ai), `local` — set at startup via `--retriever-type`.
  - **Special URLs**: `local://<docid>` fetches documents from the local tool server (always allowed regardless of retriever mode).

### Authentication

```http
X-API-Key: your_configured_api_key
```

### Environment Variables

Configure via environment or a `.env` file in the project root:

| Variable | Default | Description |
|---|---|---|
| `TOOLS_API_KEY` | — | Required API key for all requests |
| `LOCAL_SEARCH_URL` | `http://localhost:8080` | URL of the local tool server |
| `RETRIEVER_TYPE` | `custom` | Default retriever (`local`, `custom`, `jina`) |

### Run the Server

```bash
uv run tool_server/main.py --port 8001 --searcher-type ddg --retriever-type local
```

#### Command-line Arguments

| Argument | Choices | Default | Description |
|---|---|---|---|
| `--port` | — | `8001` | Port to listen on |
| `--searcher-type` | `jina`, `ddg`, `local`, `serper` | `local` | Search provider |
| `--retriever-type` | `local`, `custom`, `jina` | `local` | Web page retriever |
| `--local-search-url` | — | `http://localhost:8080` | URL of the local tool server |

### Quick usage examples

```bash
# Search
curl -H "X-API-Key: YOUR_API_KEY" "http://localhost:8001/search?q=large+language+models&num=5"

# Fetch a web page
curl -H "X-API-Key: YOUR_API_KEY" --get \
  --data-urlencode "url=https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)" \
  "http://localhost:8001/click"

# Fetch a local document
curl -H "X-API-Key: YOUR_API_KEY" "http://localhost:8001/click?url=local://doc123"
```

### Notes

- `debug=1` returns extra diagnostic info (response headers, content lengths, etc.).
- Concurrent search requests are limited to 3 at a time via an internal semaphore.
- No automatic fallbacks — if the configured provider fails, an error is returned directly.
