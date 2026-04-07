import ast
import base64
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set
from urllib.parse import quote

import requests

# =========================
# Configuration
# =========================
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "YOUR_GITHUB_TOKEN_HERE")

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}

BASE_URL = "https://api.github.com"

MIN_STARS = 10
PUSHED_WITHIN_DAYS = 365
MAX_REPOS_TO_COLLECT = 500
PER_PAGE = 100

OUTPUT_CSV  = "smell1_improper_model_reuse_results.csv"
OUTPUT_JSONL = "smell1_improper_model_reuse_results.jsonl"

# -----------------------------------------------------------------------
# Model-creation call names (Keras / TF)
# -----------------------------------------------------------------------
MODEL_CREATION_CALLS = {
    "build_model",
    "Sequential",
    "Model",
    "create_model",
    "get_model",
    "make_model",
    "define_model",
    "load_model",
    "build_generator",
    "build_discriminator",
    "tf.keras.Sequential",
    "keras.Sequential",
    "tf.keras.Model",
    "keras.Model",
}

# -----------------------------------------------------------------------
# Naming-convention patterns for custom model-builder functions.
# Any function whose name starts with a MODEL_PREFIXES token OR ends with
# a MODEL_SUFFIXES token is treated as a model-creation call.
# This covers the huge variety of custom wrapper names used in real repos.
# -----------------------------------------------------------------------
MODEL_PREFIXES = {
    "build", "create", "make", "get", "init",
    "construct", "define", "setup", "compile",
}

MODEL_SUFFIXES = {
    "model", "network", "net", "classifier",
    "generator", "discriminator", "encoder", "decoder",
    "cnn", "rnn", "lstm", "gru", "transformer",
    "autoencoder", "gan", "vae",
}

# -----------------------------------------------------------------------
# Memory-clearing calls — their PRESENCE inside a loop means NO smell
# -----------------------------------------------------------------------
CLEAR_SESSION_NAMES = {
    "clear_session",          # tf.keras.backend.clear_session()
    "reset_default_graph",    # tf.compat.v1.reset_default_graph()
}

# -----------------------------------------------------------------------
# GitHub repo search queries — extended to capture loop-based training
# -----------------------------------------------------------------------
REPO_QUERIES = [
    "tensorflow keras training loop language:Python fork:false archived:false",
    "tensorflow fit epoch language:Python fork:false archived:false",
    "topic:tensorflow language:Python fork:false archived:false",
    "topic:keras language:Python fork:false archived:false",
    "keras hyperparameter search language:Python fork:false archived:false",
]

# -----------------------------------------------------------------------
# Code search keywords for each pass
# -----------------------------------------------------------------------
LOOP_SEARCH_KEYWORDS = [
    "Sequential",
    "build_model",
    "create_model",
    "build_generator",
    "build_discriminator",
    "Model",
    "load_model",
    "make_model",
    "define_model",
]


# =========================
# Utilities
# =========================
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def pushed_after_iso(days: int) -> str:
    dt = utc_now() - timedelta(days=days)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def github_get(url: str, params: Optional[dict] = None, max_retries: int = 5) -> dict:
    """GET with rate-limit handling and exponential back-off."""
    for attempt in range(max_retries):
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code in (403, 429):
            reset     = resp.headers.get("X-RateLimit-Reset")
            remaining = resp.headers.get("X-RateLimit-Remaining")
            message   = ""
            try:
                message = resp.json().get("message", "")
            except Exception:
                pass
            if remaining == "0" and reset:
                sleep_seconds = max(int(reset) - int(time.time()) + 2, 2)
                print(f"[rate-limit] Sleeping {sleep_seconds}s")
                time.sleep(sleep_seconds)
                continue
            backoff = min(2 ** attempt, 60)
            print(f"[retry] {resp.status_code}: {message} -> sleeping {backoff}s")
            time.sleep(backoff)
            continue

        if 500 <= resp.status_code < 600:
            backoff = min(2 ** attempt, 60)
            print(f"[server-error] {resp.status_code} -> sleeping {backoff}s")
            time.sleep(backoff)
            continue

        try:
            err_json = resp.json()
        except Exception:
            err_json = {"raw": resp.text}
        raise RuntimeError(f"GitHub API error {resp.status_code}: {err_json}")

    raise RuntimeError(f"Failed after {max_retries} retries: {url}")


# =========================
# Step 1: Collect candidate repos
# =========================
def search_repositories(query: str, max_repos: int) -> List[dict]:
    items: List[dict] = []
    page = 1
    while len(items) < max_repos:
        params = {
            "q": query,
            "sort": "updated",
            "order": "desc",
            "per_page": PER_PAGE,
            "page": page,
        }
        data  = github_get(f"{BASE_URL}/search/repositories", params=params)
        batch = data.get("items", [])
        if not batch:
            break
        items.extend(batch)
        if len(batch) < PER_PAGE:
            break
        page += 1
    return items[:max_repos]


def collect_candidate_repos() -> List[dict]:
    pushed_after  = pushed_after_iso(PUSHED_WITHIN_DAYS)
    per_q_budget  = max(50, MAX_REPOS_TO_COLLECT // max(len(REPO_QUERIES), 1))
    seen: Dict[str, dict] = {}

    for base_query in REPO_QUERIES:
        full_query = f"{base_query} stars:>={MIN_STARS} pushed:>={pushed_after}"
        print(f"[repo-search] {full_query}")
        repos = search_repositories(full_query, per_q_budget)

        for repo in repos:
            full_name = repo["full_name"]
            stars     = repo.get("stargazers_count", 0)
            pushed_at = repo.get("pushed_at", "")
            if stars < MIN_STARS:
                continue
            if pushed_at and pushed_at < pushed_after:
                continue
            if full_name not in seen:
                seen[full_name] = repo
            if len(seen) >= MAX_REPOS_TO_COLLECT:
                break

        if len(seen) >= MAX_REPOS_TO_COLLECT:
            break

    result = list(seen.values())
    print(f"[repo-search] Collected {len(result)} candidate repos")
    return result


# =========================
# Step 2: Code search inside each repo
# =========================
def search_code_in_repo(repo_full_name: str) -> List[dict]:
    """
    Two-pass search:
      Pass 1 — keywords only (catches Patterns A, B, D).
      Pass 2 — del + keyword (extra coverage for Pattern B).
    Results are deduplicated by file path.
    """
    items: List[dict] = []
    existing_paths: set = set()

    def _add_batch(batch: List[dict]) -> None:
        for item in batch:
            if item["path"] not in existing_paths:
                items.append(item)
                existing_paths.add(item["path"])

    # Pass 1: model keyword — catches Patterns A, B, and D
    for keyword in LOOP_SEARCH_KEYWORDS:
        q = f'repo:{repo_full_name} "{keyword}" extension:py'
        page = 1
        while True:
            params = {"q": q, "per_page": 100, "page": page}
            try:
                data  = github_get(f"{BASE_URL}/search/code", params=params)
            except RuntimeError as e:
                print(f"  [search-warn] {e}")
                break
            batch = data.get("items", [])
            if not batch:
                break
            _add_batch(batch)
            if len(batch) < 100:
                break
            page += 1
        time.sleep(0.3)

    # Pass 2: del + model keyword — extra coverage for Pattern B
    for keyword in ("Sequential", "build_model", "create_model", "load_model"):
        q = f'repo:{repo_full_name} "del" "{keyword}" extension:py'
        page = 1
        while True:
            params = {"q": q, "per_page": 100, "page": page}
            try:
                data  = github_get(f"{BASE_URL}/search/code", params=params)
            except RuntimeError as e:
                print(f"  [search-warn] {e}")
                break
            batch = data.get("items", [])
            if not batch:
                break
            _add_batch(batch)
            if len(batch) < 100:
                break
            page += 1
        time.sleep(0.3)

    return items


def fetch_file_content(repo_full_name: str, path: str) -> Optional[str]:
    encoded_path = quote(path, safe="/")
    url  = f"{BASE_URL}/repos/{repo_full_name}/contents/{encoded_path}"
    data = github_get(url)

    if isinstance(data, list):
        return None

    if data.get("encoding") == "base64" and "content" in data:
        raw = base64.b64decode(data["content"])
        return raw.decode("utf-8", errors="replace")

    download_url = data.get("download_url")
    if download_url:
        resp = requests.get(download_url, timeout=30)
        resp.raise_for_status()
        return resp.text

    return None


# =========================
# Step 3: AST helpers
# =========================
def _call_name(node: ast.AST) -> Optional[str]:
    """
    Return a best-effort string name for a function call node.
    e.g.  build_model()         -> 'build_model'
          tf.keras.Sequential() -> 'tf.keras.Sequential'
    """
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts = []
        cur   = func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def _matches_convention(name: str) -> bool:
    """
    Step 1 — Naming-convention check.
    Returns True if the bare function name (last segment after any dots)
    matches a known prefix or suffix pattern, e.g.:
        build_resnet, create_network, get_classifier,
        init_cnn, my_encoder, setup_transformer …
    """
    bare = name.split(".")[-1].lower()
    parts = bare.split("_")
    # prefix match: first token is a known builder verb
    if parts[0] in MODEL_PREFIXES:
        return True
    # suffix match: last token is a known architecture/role noun
    if parts[-1] in MODEL_SUFFIXES:
        return True
    return False


def _is_model_call(node: ast.AST, extra: Optional[Set[str]] = None) -> bool:
    """
    Return True if node is a Call whose name:
      • matches a hardcoded MODEL_CREATION_CALLS entry, OR
      • matches a naming-convention pattern (prefix / suffix), OR
      • is in the per-file resolved set `extra` (two-pass result).
    """
    name = _call_name(node)
    if name is None:
        return False
    bare = name.split(".")[-1]
    # hardcoded list
    for mc in MODEL_CREATION_CALLS:
        if name == mc or name.endswith("." + mc):
            return True
    # naming convention
    if _matches_convention(bare):
        return True
    # per-file resolved custom functions
    if extra and (bare in extra or name in extra):
        return True
    return False


def _collect_model_returning_functions(tree: ast.AST) -> Set[str]:
    """
    Step 2 — Two-pass resolver.
    Walks all FunctionDef / AsyncFunctionDef nodes in the file.
    A function is added to the resolved set when ANY return statement inside it:
      (a) directly returns a model call  (return Sequential(...)), or
      (b) returns a variable that was assigned a model call within the same function.
    Returns a set of bare function names found to return models.
    This lets the detector recognise custom wrappers like:
        def get_classifier(): ...  return model
        def my_cnn_v2(): ...       return Sequential(...)
    even when the name does not match any convention.
    """
    resolved: Set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        func_name = node.name

        # Collect variables assigned a model call anywhere in this function
        local_model_vars: Set[str] = set()
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                tgt = stmt.targets[0]
                if isinstance(tgt, ast.Name) and _is_model_call(stmt.value):
                    local_model_vars.add(tgt.id)

        # Check every return statement
        for stmt in ast.walk(node):
            if not isinstance(stmt, ast.Return) or stmt.value is None:
                continue
            # (a) return build_model() / return Sequential(...)
            if _is_model_call(stmt.value):
                resolved.add(func_name)
                break
            # (b) return <var> where var was assigned a model call
            if isinstance(stmt.value, ast.Name) and stmt.value.id in local_model_vars:
                resolved.add(func_name)
                break

    return resolved


def _get_names_used(node: ast.AST) -> Set[str]:
    """
    Return all Name ids that appear in Load context within node.
    Used to check whether a variable is referenced between two assignments.
    """
    used: Set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            used.add(n.id)
    return used


def _contains_clear_session(node: ast.AST) -> bool:
    """
    Return True if any Call anywhere inside the subtree is a known
    memory-clearing function (clear_session, reset_default_graph).
    Their presence means the developer is already managing memory correctly.
    """
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            name = _call_name(n)
            if name:
                for cs in CLEAR_SESSION_NAMES:
                    if name == cs or name.endswith("." + cs):
                        return True
    return False


# =========================
# Step 4: Smell detection
# =========================
def detect_smell1_improper_model_reuse(source: str) -> Optional[Dict]:
    """
    Detects Smell 1 — Improper Model Reuse.

    Resolution order (per file):
      1. Hardcoded MODEL_CREATION_CALLS entries.
      2. Naming-convention patterns (build_*/create_*/*_model/*_network …).
      3. Two-pass resolver: functions found in this file that return a model.

    Pattern A — overwrite (non-use checked):
        model = build_model()
        # var never referenced here
        model = build_model()   <-- smell

    Pattern B — del + recreate:
        temp = build_model()
        del temp
        model = build_model()

    Pattern D — model creation inside a loop, no clear_session():
        for config in search_space:
            model = build_model(config)   <-- smell: graph accumulates
            model.fit(...)
            # missing: tf.keras.backend.clear_session()
    """
    lines = source.splitlines()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _detect_smell1_regex_fallback(source, lines)

    # ── Two-pass: resolve custom model-returning functions in this file ──────
    extra_model_funcs: Set[str] = _collect_model_returning_functions(tree)
    if extra_model_funcs:
        print(f"  [resolver] custom model functions found: {extra_model_funcs}")

    def is_mc(node: ast.AST) -> bool:
        """Shorthand: checks hardcoded + convention + per-file resolved."""
        return _is_model_call(node, extra=extra_model_funcs)

    LOOK_AHEAD = 30

    for parent in ast.walk(tree):
        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue

        stmts = body

        for i, stmt in enumerate(stmts):

            # ------------------------------------------------------------------
            # Pattern A: same variable overwritten — non-use check applied
            # ------------------------------------------------------------------
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, ast.Name) and is_mc(stmt.value):
                    var_name   = target.id
                    first_line = stmt.lineno
                    first_code = lines[first_line - 1].strip() if first_line <= len(lines) else ""

                    for j in range(i + 1, min(i + 1 + LOOK_AHEAD, len(stmts))):
                        s2 = stmts[j]

                        used_in_between = any(
                            var_name in _get_names_used(stmts[k])
                            for k in range(i + 1, j)
                        )
                        if used_in_between:
                            break

                        if (
                            isinstance(s2, ast.Assign)
                            and len(s2.targets) == 1
                            and isinstance(s2.targets[0], ast.Name)
                            and s2.targets[0].id == var_name
                            and is_mc(s2.value)
                        ):
                            second_line = s2.lineno
                            second_code = lines[second_line - 1].strip() if second_line <= len(lines) else ""
                            return {
                                "smell_name" : "Smell 1: Improper Model Reuse",
                                "pattern"    : "A — model variable overwritten (first instance unused)",
                                "temp_var"   : var_name,
                                "temp_line"  : first_line,
                                "temp_code"  : first_code,
                                "del_line"   : "",
                                "del_code"   : "",
                                "second_line": second_line,
                                "second_code": second_code,
                                "reason": (
                                    f"Model variable '{var_name}' is created at line {first_line} "
                                    f"and overwritten at line {second_line} without being referenced "
                                    f"in between — the first instance is wasted."
                                ),
                            }

            # ------------------------------------------------------------------
            # Pattern B: del + recreate
            # ------------------------------------------------------------------
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, ast.Name) and is_mc(stmt.value):
                    temp_var  = target.id
                    temp_line = stmt.lineno
                    temp_code = lines[temp_line - 1].strip() if temp_line <= len(lines) else ""

                    del_line = None
                    del_code = ""
                    del_idx  = None

                    for j in range(i + 1, min(i + 1 + LOOK_AHEAD, len(stmts))):
                        s2 = stmts[j]
                        if isinstance(s2, ast.Delete):
                            for td in s2.targets:
                                if isinstance(td, ast.Name) and td.id == temp_var:
                                    del_line = s2.lineno
                                    del_code = lines[del_line - 1].strip() if del_line <= len(lines) else ""
                                    del_idx  = j
                                    break
                        if del_idx is not None:
                            break

                    if del_idx is not None:
                        for k in range(del_idx + 1, min(del_idx + 1 + LOOK_AHEAD, len(stmts))):
                            s3 = stmts[k]
                            if isinstance(s3, ast.Assign) and is_mc(s3.value):
                                second_line = s3.lineno
                                second_code = lines[second_line - 1].strip() if second_line <= len(lines) else ""
                                return {
                                    "smell_name" : "Smell 1: Improper Model Reuse",
                                    "pattern"    : "B — del then recreate",
                                    "temp_var"   : temp_var,
                                    "temp_line"  : temp_line,
                                    "temp_code"  : temp_code,
                                    "del_line"   : del_line,
                                    "del_code"   : del_code,
                                    "second_line": second_line,
                                    "second_code": second_code,
                                    "reason": (
                                        f"Temporary model '{temp_var}' created at line {temp_line}, "
                                        f"deleted at line {del_line}, then a fresh model is created "
                                        f"at line {second_line} — wasteful model reuse."
                                    ),
                                }

            # ------------------------------------------------------------------
            # Pattern D: model creation inside a loop, no clear_session()
            # ------------------------------------------------------------------
            if isinstance(stmt, (ast.For, ast.While)):
                loop_line = stmt.lineno
                loop_code = lines[loop_line - 1].strip() if loop_line <= len(lines) else ""

                for loop_stmt in stmt.body:
                    if (
                        isinstance(loop_stmt, ast.Assign)
                        and len(loop_stmt.targets) == 1
                    ):
                        tgt = loop_stmt.targets[0]
                        if isinstance(tgt, ast.Name) and is_mc(loop_stmt.value):
                            model_var  = tgt.id
                            model_line = loop_stmt.lineno
                            model_code = (
                                lines[model_line - 1].strip()
                                if model_line <= len(lines) else ""
                            )
                            if not _contains_clear_session(stmt):
                                return {
                                    "smell_name" : "Smell 1: Improper Model Reuse",
                                    "pattern"    : "D — model recreated in loop without clear_session()",
                                    "temp_var"   : model_var,
                                    "temp_line"  : model_line,
                                    "temp_code"  : model_code,
                                    "del_line"   : loop_line,
                                    "del_code"   : loop_code,
                                    "second_line": "",
                                    "second_code": "",
                                    "reason": (
                                        f"Model '{model_var}' is created inside a loop at line {model_line} "
                                        f"(loop starts at line {loop_line}) without calling clear_session() "
                                        f"or reset_default_graph() — TensorFlow computation graphs and weights "
                                        f"accumulate in memory across every iteration."
                                    ),
                                }

    return None


# =========================
# Regex fallback (unchanged)
# =========================
def _detect_smell1_regex_fallback(source: str, lines: List[str]) -> Optional[Dict]:
    """
    Regex fallback for files that cannot be parsed by ast.parse.
    Covers Pattern B only (del + recreate) since it is the most structural.
    """
    create_pat = re.compile(
        r'^(\w+)\s*=\s*'
        r'(?:(?:[\w.]+\.)?)(?:'
        + "|".join(re.escape(m.split(".")[-1]) for m in MODEL_CREATION_CALLS)
        + r')\s*\(',
        re.MULTILINE,
    )
    del_pat = re.compile(r'^\s*del\s+(\w+)', re.MULTILINE)

    assignments: List[tuple] = []
    for m in create_pat.finditer(source):
        lineno = source[:m.start()].count("\n") + 1
        assignments.append((lineno, m.group(1), lines[lineno - 1].strip()))

    for a_lineno, a_var, a_code in assignments:
        for dm in del_pat.finditer(source):
            if dm.group(1) != a_var:
                continue
            d_lineno = source[:dm.start()].count("\n") + 1
            if d_lineno <= a_lineno:
                continue
            for b_lineno, b_var, b_code in assignments:
                if b_lineno <= d_lineno:
                    continue
                return {
                    "smell_name"   : "Smell 1: Improper Model Reuse",
                    "pattern"      : "B — del then recreate (regex fallback)",
                    "temp_var"     : a_var,
                    "temp_line"    : a_lineno,
                    "temp_code"    : a_code,
                    "del_line"     : d_lineno,
                    "del_code"     : lines[d_lineno - 1].strip(),
                    "second_line"  : b_lineno,
                    "second_code"  : b_code,
                    "reason"       : (
                        f"[regex] Temporary model '{a_var}' created at line {a_lineno}, "
                        f"deleted at line {d_lineno}, re-created at line {b_lineno}."
                    ),
                }
    return None


# =========================
# Step 5: Orchestrate the scan
# =========================
def scan_repo_for_smell1(repo: dict) -> List[Dict]:
    repo_full_name = repo["full_name"]
    print(f"[code-search] {repo_full_name}")

    matches         = []
    candidate_files = search_code_in_repo(repo_full_name)

    for item in candidate_files:
        path     = item["path"]
        html_url = item.get("html_url", "")
        filename = os.path.basename(path).lower()
        print(f"  [file] {path}")

        file_text = fetch_file_content(repo_full_name, path)
        if not file_text:
            continue

        finding = detect_smell1_improper_model_reuse(file_text)
        if not finding:
            continue

        record = {
            "repo_full_name"  : repo_full_name,
            "repo_html_url"   : repo.get("html_url", ""),
            "repo_description": repo.get("description", ""),
            "stars"           : repo.get("stargazers_count", 0),
            "pushed_at"       : repo.get("pushed_at", ""),
            "default_branch"  : repo.get("default_branch", ""),
            "file_path"       : path,
            "file_html_url"   : html_url,
            **finding,
        }
        matches.append(record)

    return matches


def write_outputs(results: List[Dict]) -> None:
    if not results:
        print("[output] No matches found.")
        return

    fieldnames = [
        "repo_full_name",
        "repo_html_url",
        "repo_description",
        "stars",
        "pushed_at",
        "default_branch",
        "file_path",
        "file_html_url",
        "smell_name",
        "pattern",
        "temp_var",
        "temp_line",
        "temp_code",
        "del_line",
        "del_code",
        "second_line",
        "second_code",
        "reason",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[output] Wrote {len(results)} matches to:")
    print(f"  - {OUTPUT_CSV}")
    print(f"  - {OUTPUT_JSONL}")


def main():
    if not GITHUB_TOKEN or GITHUB_TOKEN == "YOUR_GITHUB_TOKEN_HERE":
        print("ERROR: Please set your GITHUB_TOKEN before running.")
        sys.exit(1)

    # Phase 1: repository collection
    repos = collect_candidate_repos()

    # Phase 2: per-repo scanning
    all_results: List[Dict] = []
    for idx, repo in enumerate(repos, start=1):
        print(f"[scan] {idx}/{len(repos)} -> {repo['full_name']}")
        try:
            repo_results = scan_repo_for_smell1(repo)
            all_results.extend(repo_results)
        except Exception as e:
            print(f"[warn] Skipped {repo['full_name']}: {e}")

    # Phase 3: deduplication & sorting
    dedup: Dict[tuple, Dict] = {}
    for row in all_results:
        key = (
            row["repo_full_name"],
            row["file_path"],
            row.get("temp_line"),
            row.get("del_line"),
        )
        dedup[key] = row

    final_results = list(dedup.values())
    final_results.sort(
        key=lambda x: (
            -int(x.get("stars", 0)),
            x.get("repo_full_name", ""),
            x.get("file_path", ""),
        )
    )

    write_outputs(final_results)
    print(f"[done] Total verified smell-1 matches: {len(final_results)}")


if __name__ == "__main__":
    main()
