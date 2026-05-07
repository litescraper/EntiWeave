import os
import argparse
import threading
from urllib.parse import urlparse

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Header, HTTPException, Depends
from fastapi.responses import JSONResponse

from modules.search.ddg import DuckDuckGoSearcher
from modules.search.jina import JinaSearcher
from modules.search.serper import SerperSearcher
from modules.search.local import LocalSearcher
from modules.open.web import WebPageRetriever
from modules.open.jina import JinaWebReader

load_dotenv()

app = FastAPI(title="Tools API", version="1.0.0")

search_semaphore = threading.Semaphore(3)

searcher_instance = None
local_search_url_config = os.getenv("LOCAL_SEARCH_URL", "http://localhost:8000")
retriever_type_config = os.getenv("RETRIEVER_TYPE", "custom")


def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    expected = os.getenv("TOOLS_API_KEY")
    if not expected:
        raise HTTPException(status_code=500, detail="TOOLS_API_KEY not configured on server")
    if x_api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key


@app.get("/search", dependencies=[Depends(verify_api_key)])
def api_search(
    q: str = Query(...),
    num: int = Query(10),
    debug: int = Query(0),
):
    if not q.strip():
        return JSONResponse({"error": "Query 'q' cannot be empty"}, status_code=400)
    if searcher_instance is None:
        return JSONResponse({"error": "Searcher not initialized"}, status_code=500)

    with search_semaphore:
        try:
            return searcher_instance.search(q, num=num, return_debug=bool(debug))
        except Exception as e:
            return JSONResponse({"error": f"Search error: {e}"}, status_code=500)



@app.get("/click", dependencies=[Depends(verify_api_key)])
def api_click(
    url: str = Query(...),
    debug: int = Query(0),
):
    if not url.strip():
        return JSONResponse({"error": "URL cannot be empty"}, status_code=400)

    if url.startswith("local://"):
        return _fetch_local_document(url, debug=bool(debug))

    if retriever_type_config == "local":
        return JSONResponse(
            {"error": "Only local:// URLs are allowed in local retriever mode"},
            status_code=403,
        )

    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return JSONResponse({"error": f"Invalid URL format: {url}"}, status_code=400)
    except Exception:
        return JSONResponse({"error": f"Malformed URL: {url}"}, status_code=400)

    try:
        result = _fetch_remote(url, debug=bool(debug))
    except Exception as e:
        return JSONResponse({"error": f"Internal server error: {e}"}, status_code=500)

    if result and "error" in result:
        status = result.get("status", 500)
        http_status = status if isinstance(status, int) and 400 <= status < 500 else 500
        return JSONResponse(result, status_code=http_status)

    return result


def _fetch_local_document(url: str, debug: bool) -> dict:
    docid = url[len("local://"):]
    base_url = local_search_url_config.rstrip('/')
    try:
        resp = requests.post(f"{base_url}/get_document", json={"docid": docid}, timeout=30)
        resp.raise_for_status()
        doc = resp.json()
        result = {
            "url": url,
            "text": doc.get("text") or doc.get("snippet") or "No content found",
            "content_type": "text/plain",
            "provider": "local_tool_server",
        }
        if debug:
            result["debug"] = {"docid": docid, "tool_server_response": doc}
        return result
    except Exception as e:
        return JSONResponse({"error": f"Failed to fetch local document {docid}: {e}"}, status_code=500)


def _fetch_remote(url: str, debug: bool) -> dict:
    if retriever_type_config == "jina":
        return JinaWebReader().read(url, debug=debug)
    return WebPageRetriever().fetch(url, debug=debug)


def _build_searcher(searcher_type: str, local_url: str):
    if searcher_type == "jina":
        return JinaSearcher()
    if searcher_type == "serper":
        return SerperSearcher()
    if searcher_type == "ddg":
        return DuckDuckGoSearcher()
    return LocalSearcher(base_url=local_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tools API Server")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--searcher-type", choices=["jina", "ddg", "local", "serper"], default="local")
    parser.add_argument("--retriever-type", choices=["local", "custom", "jina"], default="local")
    parser.add_argument("--local-search-url", default=os.getenv("LOCAL_SEARCH_URL", "http://localhost:8080"))
    args = parser.parse_args()

    local_search_url_config = args.local_search_url
    retriever_type_config = args.retriever_type
    searcher_instance = _build_searcher(args.searcher_type, args.local_search_url)

    print(f"Starting Tools API with {args.searcher_type} searcher and {args.retriever_type} retriever on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
