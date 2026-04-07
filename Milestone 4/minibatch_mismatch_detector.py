"""
Smell 2: Minibatch Mismatch Detector
Mines GitHub repositories for excessively large batch sizes in TensorFlow/Keras code.

Definition: Minibatch Mismatch occurs when excessively large minibatch sizes (>= 512)
are used, increasing memory demand and potentially degrading training stability.

Adapted from professor's smell1_detector_v3.py structure.

Usage:
    python smell2_detector.py
"""

import ast
import base64
import csv
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
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "*******************")

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}

BASE_URL = "https://api.github.com"

MIN_STARS = 10
PUSHED_WITHIN_DAYS = 365
MAX_REPOS_TO_COLLECT = 3000
PER_PAGE = 100

BATCH_SIZE_THRESHOLD = 512

OUTPUT_CSV = "smell2_results.csv"

# =========================
# Repo search queries
# =========================
REPO_QUERIES = [
    "tensorflow keras training language:Python fork:false archived:false",
    "tensorflow fit batch_size language:Python fork:false archived:false",
    "topic:tensorflow language:Python fork:false archived:false",
    "topic:keras language:Python fork:false archived:false",
    "keras model training language:Python fork:false archived:false",
    "tensorflow deep learning language:Python fork:false archived:false",
    "keras CNN classifier language:Python fork:false archived:false",
    "tensorflow image classification language:Python fork:false archived:false",
]

# Code search keywords to find relevant files inside repos
CODE_SEARCH_KEYWORDS = [
    "batch_size",
    "BATCH_SIZE",
]

# =========================
# Utilities
# =========================

def utc_now():
    return datetime.now(timezone.utc)


def pushed_after_iso(days):
    dt = utc_now() - timedelta(days=days)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def github_get(url, params=None, max_retries=5):
    """GET with rate-limit handling and exponential back-off."""
    for attempt in range(max_retries):
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code in (403, 429):
            reset     = resp.headers.get("X-RateLimit-Remaining")
            remaining = resp.headers.get("X-RateLimit-Remaining")
            message   = ""
            try:
                message = resp.json().get("message", "")
            except Exception:
                pass
            if remaining == "0" and reset:
                sleep_seconds = max(int(resp.headers.get("X-RateLimit-Reset", time.time() + 60)) - int(time.time()) + 2, 2)
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

def search_repositories(query, max_repos):
    items = []
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
        time.sleep(1)
    return items[:max_repos]


def collect_candidate_repos():
    pushed_after = pushed_after_iso(PUSHED_WITHIN_DAYS)
    per_q_budget = max(50, MAX_REPOS_TO_COLLECT // max(len(REPO_QUERIES), 1))
    seen = {}

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

        time.sleep(2)

    result = list(seen.values())
    print(f"[repo-search] Collected {len(result)} candidate repos")
    return result


# =========================
# Step 2: Code search inside each repo
# =========================

def search_code_in_repo(repo_full_name):
    """Search for files containing batch_size keywords inside a repo."""
    items = []
    existing_paths = set()

    def _add_batch(batch):
        for item in batch:
            if item["path"] not in existing_paths:
                items.append(item)
                existing_paths.add(item["path"])

    for keyword in CODE_SEARCH_KEYWORDS:
        q = f'repo:{repo_full_name} "{keyword}" extension:py'
        page = 1
        while True:
            params = {"q": q, "per_page": 100, "page": page}
            try:
                data = github_get(f"{BASE_URL}/search/code", params=params)
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
        time.sleep(0.5)

    return items


def fetch_file_content(repo_full_name, path):
    """Fetch file content from GitHub."""
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
# Step 3: Smell detection
# =========================

PATTERNS = [
    re.compile(r'batch[_\-\s]*size\s*=\s*(\d+)', re.IGNORECASE),
    re.compile(r'batch[_\-\s]*size["\'].*?default\s*=\s*(\d+)', re.IGNORECASE),
    re.compile(r'\.fit\s*\(.*?batch_size\s*=\s*(\d+)', re.IGNORECASE | re.DOTALL),
    re.compile(r'DataLoader\s*\(.*?batch_size\s*=\s*(\d+)', re.IGNORECASE | re.DOTALL),
    re.compile(r'BATCH[_\-]*SIZE\s*=\s*(\d+)', re.IGNORECASE),
]


def detect_smell2_minibatch_mismatch(source):
    """
    Detect Minibatch Mismatch: batch_size >= 512.
    Returns list of findings (one per offending line).
    """
    findings = []
    for line_num, line in enumerate(source.splitlines(), start=1):
        for pattern in PATTERNS:
            match = pattern.search(line)
            if match:
                try:
                    value = int(match.group(1))
                except ValueError:
                    continue
                if value >= BATCH_SIZE_THRESHOLD:
                    findings.append({
                        "line_number":  line_num,
                        "code_snippet": line.strip()[:200],
                        "batch_value":  value,
                        "explanation": (
                            f"Minibatch Mismatch: batch_size={value} exceeds threshold of "
                            f"{BATCH_SIZE_THRESHOLD}. Large batch sizes increase memory pressure, "
                            f"can reduce model generalization, and waste computational resources."
                        ),
                    })
                    break
    return findings


# =========================
# Step 4: Scan repo
# =========================

def scan_repo_for_smell2(repo):
    repo_full_name = repo["full_name"]
    print(f"[scan] {repo_full_name}")

    matches         = []
    candidate_files = search_code_in_repo(repo_full_name)

    for item in candidate_files:
        path     = item["path"]
        html_url = item.get("html_url", "")
        print(f"  [file] {path}")

        file_text = fetch_file_content(repo_full_name, path)
        if not file_text:
            continue

        findings = detect_smell2_minibatch_mismatch(file_text)
        for finding in findings:
            matches.append({
                "repository_name": repo_full_name,
                "repository_url":  repo.get("html_url", ""),
                "file_path":       path,
                "file_url":        html_url,
                "stars":           repo.get("stargazers_count", 0),
                "last_updated":    repo.get("pushed_at", ""),
                "line_number":     finding["line_number"],
                "code_snippet":    finding["code_snippet"],
                "batch_size_value": finding["batch_value"],
                "explanation":     finding["explanation"],
            })
            print(f"    SMELL line {finding['line_number']}: batch_size={finding['batch_value']}")

    return matches


# =========================
# Step 5: Write output
# =========================

FIELDNAMES = [
    "repository_name", "repository_url", "file_path", "file_url",
    "stars", "last_updated", "line_number", "code_snippet",
    "batch_size_value", "explanation",
]


def load_already_scanned():
    if not os.path.exists(OUTPUT_CSV):
        return set()
    already = set()
    with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            already.add(row["repository_name"])
    return already


# =========================
# Main
# =========================

def main():
    if not GITHUB_TOKEN or GITHUB_TOKEN == "YOUR_GITHUB_TOKEN_HERE":
        print("ERROR: Please set your GITHUB_TOKEN before running.")
        sys.exit(1)

    print("=" * 60)
    print("Smell 2: Minibatch Mismatch — GitHub Detector")
    print(f"Threshold  : batch_size >= {BATCH_SIZE_THRESHOLD}")
    print(f"Min stars  : {MIN_STARS}")
    print(f"Max repos  : {MAX_REPOS_TO_COLLECT}")
    print(f"Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load already scanned repos
    already_scanned = load_already_scanned()
    print(f"\nAlready in CSV: {len(already_scanned)} repos — will skip them")

    # Phase 1: collect repos
    repos = collect_candidate_repos()

    # Skip already scanned
    repos = [r for r in repos if r["full_name"] not in already_scanned]
    print(f"\nNew repos to scan: {len(repos)}")

    # Open CSV in append mode
    file_exists = os.path.exists(OUTPUT_CSV)
    csvfile = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer  = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()
    csvfile.flush()

    # Phase 2: scan each repo
    total = 0
    for idx, repo in enumerate(repos, start=1):
        print(f"\n[{idx}/{len(repos)}] {repo['full_name']}")
        try:
            results = scan_repo_for_smell2(repo)
            for row in results:
                writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
                csvfile.flush()
                total += 1
        except Exception as e:
            print(f"[warn] Skipped {repo['full_name']}: {e}")

    csvfile.close()

    print(f"\n{'=' * 60}")
    print(f"DONE")
    print(f"New findings : {total}")
    print(f"Saved to     : {OUTPUT_CSV}")
    print(f"Finished     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
