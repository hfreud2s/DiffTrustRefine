"""
Submits HumanEvalComm baseline batches to the OpenRouter Batch API and retrieves their results.

Step 2: submission and retrieval.
  create_batch()    - submit one category's request file; returns the batch object (with its id)
  save_batch_meta() - record the batch id and state next to the request file
  get_batch()       - fetch a batch's current state
  wait_for_batch()  - poll until the batch reaches a terminal state
  save_results()    - write a completed batch's inline results to a .jsonl (input for step 3)
  submit_llm()      - submit every not-yet-submitted category for one LLM (throttled)
  pending_categories() / retry_llm() - find and resubmit categories a rate limit skipped

Requires the OPENROUTER_API_KEY environment variable.
OpenRouter Batch API docs: https://openrouter.ai/docs/batch-quickstart
"""
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path

THIS_DIR   = Path(__file__).resolve().parent
EXPERIMENT = THIS_DIR / ".HEC-experiment"

API_BASE = "https://openrouter.ai/api/beta/batches"
ENDPOINT = "/v1/chat/completions"
TERMINAL = {"completed", "failed", "expired", "cancelled"}


class OpenRouterError(RuntimeError):
    """An error response from OpenRouter. `code` is the HTTP status (429 == rate limited)."""
    def __init__(self, code, detail, method="", url=""):
        self.code = code
        self.detail = detail
        super().__init__(f"OpenRouter {method} {url} failed: HTTP {code}\n{detail}")


def _api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("Set the OPENROUTER_API_KEY environment variable before submitting.")
    return key


def _request(url: str, method: str, payload: dict = None):
    """POST/GET against the OpenRouter API, surfacing the error body on failure."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Authorization": f"Bearer {_api_key()}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise OpenRouterError(e.code, detail, method, url) from None


def read_requests(request_path: Path):
    """Reads a build_batch .jsonl file into a list of {custom_id, body} items."""
    with open(request_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def create_batch(request_path: Path, model: str = None, endpoint: str = ENDPOINT):
    """
    Submits one category's request file as a single OpenRouter batch.

    request_path: a baseline_batch_request.jsonl produced by build_batch.py
    model:        OpenRouter model slug for the batch; if None, taken from the first request's body
    endpoint:     the API shape (default /v1/chat/completions)
    returns:      the batch object OpenRouter returns (carries the batch id and status)
    """
    requests_list = read_requests(request_path)
    if not requests_list:
        raise ValueError(f"No requests in {request_path}")
    if model is None:
        model = requests_list[0]["body"]["model"]
    payload = {"endpoint": endpoint, "model": model, "requests": requests_list}
    batch = _request(API_BASE, "POST", payload)
    print(f"submitted {len(requests_list)} requests from {request_path}")
    print(f"  batch id: {batch.get('id')}  status: {batch.get('status')}  model: {model}")
    return batch


def save_batch_meta(batch: dict, request_path: Path):
    """Writes batch_meta.json next to the request file, recording the id and submission state."""
    meta_path = request_path.parent / "batch_meta.json"
    meta = {
        "batch_id":       batch.get("id"),
        "status":         batch.get("status"),
        "model":          batch.get("model"),
        "endpoint":       batch.get("endpoint"),
        "created_at":     batch.get("created_at"),
        "request_counts": batch.get("request_counts"),
        "request_file":   request_path.name,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  meta -> {meta_path}")
    return meta_path


def get_batch(batch_id: str):
    """Fetches the current state of a batch (status, request_counts, and results when completed)."""
    return _request(f"{API_BASE}/{batch_id}", "GET")


def wait_for_batch(batch_id: str, poll: int = 60):
    """
    Polls a batch until it reaches a terminal state (completed/failed/expired/cancelled).
    Prints status and progress each poll. Returns the final batch object.
    """
    while True:
        batch  = get_batch(batch_id)
        counts = batch.get("request_counts") or {}
        print(f"{batch_id}: {batch.get('status')} "
              f"({counts.get('completed', 0)}/{counts.get('total', 0)} done, "
              f"{counts.get('failed', 0)} failed)")
        if batch.get("status") in TERMINAL:
            return batch
        time.sleep(poll)


def save_results(batch: dict, out_path: Path):
    """
    Writes a completed batch's inline results to out_path as .jsonl, one result item per line
    ({custom_id, response, error}). This is the file the post-processing step (step 3) will read.
    """
    results = batch.get("results")
    if results is None:
        raise ValueError(f"Batch {batch.get('id')} has no results yet (status={batch.get('status')})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    print(f"wrote {len(results)} results to {out_path}")
    return out_path


def count_requests(request_path: Path):
    """Number of request lines in a batch request file."""
    with open(request_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def is_submitted(category_dir: Path):
    """True if this category already has a batch_meta.json carrying a batch_id (i.e. it was submitted)."""
    meta = category_dir / "batch_meta.json"
    if not meta.exists():
        return False
    try:
        with open(meta, "r", encoding="utf-8") as f:
            return bool(json.load(f).get("batch_id"))
    except Exception:
        return False


def pending_categories(llm_dir: str):
    """
    Categories of one LLM that still need submitting: those with a baseline_batch_request.jsonl but
    no successful submission (no batch_meta.json with a batch_id). This is what a retry resubmits.

    llm_dir: e.g. "LLM1"
    returns: sorted list of category names
    """
    llm_path = EXPERIMENT / llm_dir
    return sorted(req.parent.name
                  for req in llm_path.glob("*/baseline_batch_request.jsonl")
                  if not is_submitted(req.parent))


def _submit_categories(llm_dir: str, categories: list,
                       max_requests_per_min: int = 20000, max_429_retries: int = 5, backoff: int = 60):
    """
    Submits the given categories of one LLM, keeping the number of requests sent within any 60s
    window under max_requests_per_min, and backing off on any 429 that still slips through. Saves a
    batch_meta.json per submitted category. Shared by submit_llm and retry_llm.
    """
    llm_path     = EXPERIMENT / llm_dir
    submitted    = {}
    window_start = time.time()
    used         = 0
    for category in categories:
        request_path = llm_path / category / "baseline_batch_request.jsonl"
        if not request_path.exists():
            print(f"skip {llm_dir}/{category}: no request file")
            continue
        n = count_requests(request_path)

        if time.time() - window_start >= 60:
            window_start, used = time.time(), 0
        if used and used + n > max_requests_per_min:
            wait = max(0.0, 60 - (time.time() - window_start))
            if wait:
                print(f"rate cap: waiting {wait:.0f}s before {category} "
                      f"({used} requests this minute, +{n} would exceed {max_requests_per_min})")
                time.sleep(wait)
            window_start, used = time.time(), 0

        for attempt in range(max_429_retries + 1):
            try:
                batch = create_batch(request_path)
                break
            except OpenRouterError as e:
                if e.code == 429 and attempt < max_429_retries:
                    print(f"429 on {llm_dir}/{category}; backing off {backoff}s "
                          f"(attempt {attempt + 1}/{max_429_retries})")
                    time.sleep(backoff)
                    window_start, used = time.time(), 0
                    continue
                raise
        save_batch_meta(batch, request_path)
        submitted[category] = batch.get("id")
        used += n
    return submitted


def submit_llm(llm_dir: str, categories: list = None,
               max_requests_per_min: int = 20000, resubmit: bool = False):
    """
    Submits the baseline batches for one LLM, one batch per category, throttled to stay under the
    per-minute request limit. Categories already submitted (they have a batch_meta.json) are skipped
    unless resubmit=True, so this is safe to re-run.

    llm_dir:              e.g. "LLM1"
    categories:           which categories (default: all that have a request file)
    max_requests_per_min: per-minute request cap to stay under (OpenRouter limit)
    resubmit:             if True, submit even categories that already have a batch_meta.json
    returns:              dict category -> batch_id (only the ones submitted this call)
    """
    llm_path = EXPERIMENT / llm_dir
    if categories is None:
        categories = sorted(p.parent.name for p in llm_path.glob("*/baseline_batch_request.jsonl"))
    if not resubmit:
        categories = [c for c in categories if not is_submitted(llm_path / c)]
    submitted = _submit_categories(llm_dir, categories, max_requests_per_min)
    print(f"\nsubmitted {len(submitted)} batch(es) for {llm_dir}: {submitted}")
    return submitted


def retry_llm(llm_dir: str, max_requests_per_min: int = 20000):
    """
    Resubmits only the categories of one LLM that have not been submitted yet, i.e. those with a
    request file but no batch_meta.json. Use this after a submission cut short by the per-minute
    request limit: it reports what is pending, then submits it under the same throttle.

    llm_dir:              e.g. "LLM1"
    max_requests_per_min: per-minute request cap to stay under
    returns:              dict category -> batch_id
    """
    pending = pending_categories(llm_dir)
    if not pending:
        print(f"{llm_dir}: nothing to retry, every category has a batch_meta.json")
        return {}
    counts = {c: count_requests(EXPERIMENT / llm_dir / c / "baseline_batch_request.jsonl") for c in pending}
    print(f"{llm_dir}: {len(pending)} pending ("
          + ", ".join(f"{c}={counts[c]}" for c in pending)
          + f", {sum(counts.values())} requests total)")
    submitted = _submit_categories(llm_dir, pending, max_requests_per_min)
    print(f"\nresubmitted {len(submitted)} batch(es) for {llm_dir}: {submitted}")
    return submitted


if __name__ == "__main__":

    # --- First submission for an LLM (skips anything already submitted, throttled) ---
    # submit_llm("LLM1")

    # --- Retry: resubmit only the categories a rate limit skipped ---
    # retry_llm("LLM1")

    # --- Check what is still pending, without submitting ---
    # print(pending_categories("LLM1"))

    # --- Later: wait for a batch and save its results ---
    # batch_id = json.load(open(EXPERIMENT / "LLM1" / "original" / "batch_meta.json"))["batch_id"]
    # batch    = wait_for_batch(batch_id, poll=60)
    # save_results(batch, EXPERIMENT / "LLM1" / "original" / "baseline_batch_result.jsonl")
    pass
