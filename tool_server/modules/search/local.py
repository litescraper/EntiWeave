import os
import requests
from typing import Optional
import string
import re
from dataclasses import dataclass


_SENT_SPLIT = re.compile(
    r"(?<=[.!?])\s+"
    r"|(?<=\n)\s*"
    r"|(?<=[;:])\s{2,}"
)


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Split into (sentence, char_start, char_end). Merge short orphans."""
    raw_spans: list[tuple[int, int]] = []
    pos = 0
    for part in _SENT_SPLIT.split(text):
        start = text.find(part, pos)
        end = start + len(part)
        raw_spans.append((start, end))
        pos = end

    merged: list[tuple[str, int, int]] = []
    for start, end in raw_spans:
        sent = text[start:end].strip()
        if not sent:
            continue
        if len(sent.split()) < 4 and merged:
            prev_sent, prev_start, _ = merged[-1]
            merged[-1] = (prev_sent + " " + sent, prev_start, end)
        else:
            merged.append((sent, start, end))
    return merged


def normalize_snippet(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def token_positions(text: str, terms: set[str]) -> dict[str, list[int]]:
    positions: dict[str, list[int]] = {t: [] for t in terms}
    for i, tok in enumerate(tokenize(text)):
        if tok in positions:
            positions[tok].append(i)
    return positions


def phrase_match_bonus(sentence: str, query: str) -> float:
    """Bonus for consecutive query substrings appearing verbatim."""
    q_tokens = tokenize(query)
    s_lower = sentence.lower()
    bonus = 0.0
    for length in range(2, len(q_tokens) + 1):
        for start in range(len(q_tokens) - length + 1):
            phrase = " ".join(q_tokens[start: start + length])
            if phrase in s_lower:
                bonus += length ** 1.5
    return bonus


def proximity_score(positions: dict[str, list[int]]) -> float:
    """Reward sentences where matched query terms appear close together."""
    matched = [p for p in positions.values() if p]
    if len(matched) < 2:
        return 0.0
    score = 0.0
    for i in range(len(matched)):
        for j in range(i + 1, len(matched)):
            min_dist = min(abs(a - b) for a in matched[i] for b in matched[j])
            score += 1.0 / (min_dist + 1)
    return score


def coverage_score(positions: dict[str, list[int]], query_terms: set[str]) -> float:
    matched = sum(1 for p in positions.values() if p)
    return matched / max(len(query_terms), 1)


def term_freq_score(positions: dict[str, list[int]]) -> float:
    return sum(min(len(p), 2) for p in positions.values())


@dataclass
class ScoredSentence:
    text: str
    char_start: int
    char_end: int
    score: float
    idx: int


def score_sentence(
    sentence: str,
    char_start: int,
    char_end: int,
    idx: int,
    query: str,
    query_terms: set[str],
    total_sentences: int,
    weights: dict,
) -> ScoredSentence:
    positions = token_positions(sentence, query_terms)

    cov = coverage_score(positions, query_terms)
    tf = term_freq_score(positions)
    prox = proximity_score(positions)
    phrase = phrase_match_bonus(sentence, query)

    pos_bias = weights["position_bias"] * max(
        0, 1 - (idx / max(total_sentences * 0.2, 1))
    )

    word_count = len(sentence.split())
    if word_count < 8:
        length_factor = 0.5
    elif word_count > 80:
        length_factor = 0.8
    else:
        length_factor = 1.0

    score = (
        weights["coverage"] * cov
        + weights["tf"] * tf
        + weights["proximity"] * prox
        + weights["phrase"] * phrase
        + pos_bias
    ) * length_factor

    return ScoredSentence(sentence, char_start, char_end, score, idx)


def merge_adjacent(
    seed: ScoredSentence,
    scored_by_idx: dict[int, ScoredSentence],
    n_sentences: int,
    merge_threshold: float,
    max_merge: int,
) -> ScoredSentence:
    """Absorb neighbours into the seed if they exceed merge_threshold."""
    candidates = [seed]
    for direction in (-1, 1):
        for step in range(1, max_merge + 1):
            ni = seed.idx + direction * step
            if not (0 <= ni < n_sentences):
                break
            nb = scored_by_idx.get(ni)
            if nb is None or nb.score < merge_threshold:
                break
            candidates.append(nb)

    candidates.sort(key=lambda s: s.idx)
    return ScoredSentence(
        text=" ".join(s.text for s in candidates),
        char_start=candidates[0].char_start,
        char_end=candidates[-1].char_end,
        score=sum(s.score for s in candidates) / len(candidates),
        idx=candidates[0].idx,
    )


DEFAULT_WEIGHTS = {
    "coverage": 3.0,
    "tf": 1.0,
    "proximity": 2.0,
    "phrase": 4.0,
    "position_bias": 0.5,
}


def extract_snippet(
    document: str,
    query: str,
    top_n: int = 1,
    max_merge: int = 2,
    merge_threshold_ratio: float = 0.4,
    weights: Optional[dict] = None,
) -> list[str]:
    """
    Return the top-n most query-relevant snippets from a document.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    query_terms = set(tokenize(query))
    if not query_terms:
        return [normalize_snippet(document[:400])]

    sentences = split_sentences(document)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [normalize_snippet(sentences[0][0])]

    scored_all: list[ScoredSentence] = [
        score_sentence(sent, cs, ce, idx, query, query_terms, len(sentences), w)
        for idx, (sent, cs, ce) in enumerate(sentences)
    ]

    scored_by_idx = {s.idx: s for s in scored_all}
    sorted_by_score = sorted(scored_all, key=lambda s: -s.score)

    selected_seeds: list[ScoredSentence] = []
    for candidate in sorted_by_score:
        if all(abs(candidate.idx - s.idx) > max_merge for s in selected_seeds):
            selected_seeds.append(candidate)
        if len(selected_seeds) == top_n:
            break

    snippets_with_pos: list[tuple[int, str]] = []
    for seed in selected_seeds:
        threshold = seed.score * merge_threshold_ratio
        merged = merge_adjacent(seed, scored_by_idx, len(sentences), threshold, max_merge)
        prefix = "" if merged.char_start > 0 else ""
        suffix = "" if merged.char_end < len(document) else ""
        snippets_with_pos.append((merged.idx, normalize_snippet(f"{prefix}{merged.text.strip()}{suffix}")))

    snippets_with_pos.sort(key=lambda x: x[0])
    return [s for _, s in snippets_with_pos]


def extract_snippets_bulk(
    results: list[dict],
    query: str,
    doc_key: str = "content",
    top_n: int = 10,
    max_merge: int = 2,
    merge_threshold_ratio: float = 0.4,
    weights: Optional[dict] = None,
) -> list[dict]:
    """
    Process the top-k results from a search server.
    """
    return [
        {
            **result,
            "snippets": extract_snippet(
                result.get(doc_key, ""),
                query,
                top_n=top_n,
                max_merge=max_merge,
                merge_threshold_ratio=merge_threshold_ratio,
                weights=weights,
            ),
        }
        for result in results
    ]

class LocalSearcher:
    """
    A searcher that uses our local tool_server.
    """
    def __init__(self, base_url: str = None, timeout: int = 60):
        """
        Initialize the Local searcher.
        """
        self.base_url = (base_url or os.getenv("LOCAL_SEARCH_URL", "http://localhost:8000")).rstrip('/')
        self.timeout = timeout

    def search(self, query: str, num: Optional[int] = None, return_debug: bool = False) -> dict:
        """
        Perform a search using the local tool_server.
        """
        query = query.strip()
        if not query:
            return {"query": query, "results": [], "provider": "local"}

        try:
            # remove any punctuation from the query
            cleaned_query = query.translate(str.maketrans('', '', string.punctuation))
            payload = {"query": cleaned_query}
            if num is not None:
                payload["k"] = num

            resp = requests.post(f"{self.base_url}/search", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            
            results = []
            for item in data.get("results", []):
                doc_text = item.get("content") or item.get("snippet", "")
                snippets = extract_snippet(
                    doc_text,
                    cleaned_query,
                    top_n=10,
                    max_merge=2,
                    merge_threshold_ratio=0.4,
                )

                # Mapping tool_server's docid and snippet to SearchResult's title/url and description
    
                # regex to extract "title: {some title}" from the content, if exists
                title_match = re.search(r"title:\s*(.+)", doc_text, re.IGNORECASE)
                title = title_match.group(1).strip() if title_match else item.get("docid", "No Title")

                results.append({
                    "title": title,
                    "url": f"local://{item.get('docid', '')}",
                    "description": "...".join(snippets) if snippets else doc_text[:500],
                })

            output = {
                "query": query,
                "results": results,
                "provider": "local"
            }
            
            if return_debug:
                output["debug"] = {
                    "status_code": resp.status_code,
                    "target_url": resp.url
                }
            
            return output

        except Exception as e:
            # Re-raise to match searcher behavior in this project
            raise e
