import os
import re
import time
import base64
import requests
import pandas as pd
from dotenv import load_dotenv

# Load GitHub token from .env file
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Multiple search queries to maximize coverage
# Each query targets a different variation of the smell
SEARCH_QUERIES = [
    "TF_NUM_INTRAOP_THREADS tensorflow language:python",
    "TF_NUM_INTEROP_THREADS tensorflow language:python",
    "TF_NUM_INTRAOP_THREADS keras language:python",
    "TF_NUM_INTEROP_THREADS keras language:python",
    "set_intra_op_parallelism_threads tensorflow language:python",
    "set_inter_op_parallelism_threads tensorflow language:python",
    "intra_op_parallelism_threads tensorflow language:python",
    "inter_op_parallelism_threads tensorflow language:python",
    "set_intra_op_parallelism_threads keras language:python",
    "set_inter_op_parallelism_threads keras language:python",
]

# TensorFlow/Keras presence patterns
# File must import TensorFlow or Keras to be considered
TENSORFLOW_PATTERNS = [
    r'import tensorflow',
    r'import tensorflow as tf',
    r'from tensorflow',
    r'import keras',
    r'from keras',
    r'from tf\.keras',
]

# Two separate CSV files:
# 1. Active smells - lines that actually execute with value > 1
# 2. Commented smells - lines that are commented out (bad code hygiene)
ACTIVE_CSV = "unnecessary_parallelism_results.csv"
COMMENTED_CSV = "unnecessary_parallelism_commented_results.csv"


def load_existing_results():
    """
    Load previously saved results from both CSV files.
    This allows the script to resume from where it left off
    if it was interrupted by a connection error.
    """
    active_results = []
    commented_results = []
    seen_active = set()
    seen_commented = set()

    if os.path.exists(ACTIVE_CSV):
        df = pd.read_csv(ACTIVE_CSV)
        active_results = df.to_dict("records")
        seen_active = set(zip(df["repository_name"], df["file_path"], df["line_number"]))
        print(f"Loaded {len(active_results)} existing active smells")

    if os.path.exists(COMMENTED_CSV):
        df = pd.read_csv(COMMENTED_CSV)
        commented_results = df.to_dict("records")
        seen_commented = set(zip(df["repository_name"], df["file_path"], df["line_number"]))
        print(f"Loaded {len(commented_results)} existing commented smells")

    return active_results, commented_results, seen_active, seen_commented


def save_active_result(result, results):
    """
    Save an active smell result immediately to the main CSV.
    Active smells are lines that actually execute with
    a hardcoded thread value greater than 1.
    These are our primary research results.
    """
    results.append(result)
    df = pd.DataFrame(results)
    df.drop_duplicates(
        subset=["repository_name", "file_path", "line_number"],
        inplace=True
    )
    df.to_csv(ACTIVE_CSV, index=False)


def save_commented_result(result, results):
    """
    Save a commented smell result to the secondary CSV.
    Commented smells are lines starting with # that do not execute.
    These serve as evidence of bad code hygiene.
    """
    results.append(result)
    df = pd.DataFrame(results)
    df.drop_duplicates(
        subset=["repository_name", "file_path", "line_number"],
        inplace=True
    )
    df.to_csv(COMMENTED_CSV, index=False)


def check_rate_limit():
    """
    Check remaining GitHub API requests.
    GitHub allows 30 search requests per minute
    and 5000 core requests per hour with a token.
    """
    try:
        response = requests.get(
            "https://api.github.com/rate_limit",
            headers=HEADERS,
            timeout=10
        )
        data = response.json()
        search_remaining = data["resources"]["search"]["remaining"]
        core_remaining = data["resources"]["core"]["remaining"]
        print(f"Search API remaining: {search_remaining}")
        print(f"Core API remaining: {core_remaining}")
        return search_remaining, core_remaining
    except Exception as e:
        print(f"Could not check rate limit: {e}")
        return 30, 5000


def wait_if_needed(remaining, threshold=5):
    """Wait if we are running low on API requests."""
    if remaining < threshold:
        print("Rate limit low! Waiting 60 seconds...")
        time.sleep(60)


def safe_request(url, params=None, retries=3):
    """
    Make a GitHub API request with retry logic.
    If connection drops we try up to 3 times before giving up.
    This prevents the script from crashing on temporary
    network interruptions.
    """
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                headers=HEADERS,
                params=params,
                timeout=15
            )
            return response
        except Exception as e:
            print(f"Connection error (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                print("Waiting 10 seconds before retrying...")
                time.sleep(10)
    return None


def search_code(query, page=1):
    """
    Search GitHub code for a specific query.
    Returns up to 100 results per page.
    We paginate through up to 10 pages giving
    a maximum of 1000 results per query.
    """
    url = "https://api.github.com/search/code"
    params = {
        "q": query,
        "per_page": 100,
        "page": page
    }
    response = safe_request(url, params=params)
    time.sleep(3)

    if response is None:
        return [], 0
    if response.status_code == 200:
        data = response.json()
        return data.get("items", []), data.get("total_count", 0)
    elif response.status_code == 403:
        print("Rate limit hit! Waiting 60 seconds...")
        time.sleep(60)
        return [], 0
    elif response.status_code == 422:
        print(f"Query too complex, skipping: {query}")
        return [], 0
    else:
        print(f"Search failed with status: {response.status_code}")
        return [], 0


def get_repo_details(repo_full_name):
    """Fetch repository details to apply our quality filters."""
    url = f"https://api.github.com/repos/{repo_full_name}"
    response = safe_request(url)
    time.sleep(1)
    if response and response.status_code == 200:
        return response.json()
    return None


def get_file_content(url):
    """Download the raw content of a file from GitHub."""
    response = safe_request(url)
    time.sleep(1)
    if response and response.status_code == 200:
        return response.json().get("content", "")
    return ""


def passes_filters(repo_details):
    """
    Apply repository quality filters.
    We only keep repositories that are:
    - Written in Python
    - Have at least 10 stars
    - Not archived or disabled
    """
    if not repo_details:
        return False
    return (
        repo_details.get("stargazers_count", 0) >= 10 and
        not repo_details.get("archived", True) and
        not repo_details.get("disabled", True) and
        repo_details.get("language") == "Python"
    )


def is_tensorflow_file(content):
    """
    Verify the file actually uses TensorFlow or Keras.
    This keeps our results consistent with M1-M3 framework.
    """
    for pattern in TENSORFLOW_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return True
    return False


def is_commented_line(line):
    """
    Check if a line is commented out.
    A commented line starts with # ignoring leading whitespace.
    """
    return line.strip().startswith("#")


def extract_thread_value(line):
    """
    Extract the hardcoded thread number from a line of code.
    Returns the number if found, None otherwise.
    We use this to apply our detection rule:
    only flag values greater than 1.
    """
    # Match any number in the line
    match = re.search(r'[=\(]\s*["\']?(\d+)["\']?', line)
    if match:
        return int(match.group(1))
    return None


def is_smell(line):
    """
    Check if a line contains the Unnecessary Parallelism smell.
    Our detection rule:
    - Must contain a thread-related pattern
    - Must have a hardcoded number greater than 1
    Setting to 1 means no parallelism which is a different problem.
    Setting to any value > 1 bypasses TensorFlow's automatic
    resource detection and can cause resource contention.
    """
    # Thread related patterns to detect
    thread_patterns = [
        r'TF_NUM_INTRAOP_THREADS',
        r'TF_NUM_INTEROP_THREADS',
        r'set_intra_op_parallelism_threads',
        r'set_inter_op_parallelism_threads',
        r'intra_op_parallelism_threads',
        r'inter_op_parallelism_threads',
    ]

    # Check if line contains any thread pattern
    has_pattern = any(re.search(p, line) for p in thread_patterns)
    if not has_pattern:
        return False

    # Extract the hardcoded value
    value = extract_thread_value(line)

    # Only flag if value is greater than 1
    if value is not None and value > 1:
        return True

    return False


def generate_explanation(snippet, is_commented=False):
    """
    Generate a human readable explanation of why
    this line of code is considered a smell.
    """
    prefix = "COMMENTED - Bad Code Hygiene: " if is_commented else ""

    if "INTRAOP" in snippet.upper() or "intra_op" in snippet:
        return (
            f"{prefix}Unnecessary Parallelism: A hardcoded thread value "
            "greater than 1 was set for intra-op parallelism. "
            "This controls how many threads work inside a single operation. "
            "If this number exceeds the available CPU cores, threads compete "
            "for resources causing overhead instead of speedup — "
            "like having more chefs than stoves in a kitchen."
        )
    elif "INTEROP" in snippet.upper() or "inter_op" in snippet:
        return (
            f"{prefix}Unnecessary Parallelism: A hardcoded thread value "
            "greater than 1 was set for inter-op parallelism. "
            "This controls how many operations run in parallel. "
            "If this number exceeds the available CPU cores, the system "
            "wastes time context-switching between threads instead of "
            "doing useful work — defeating the purpose of parallelism."
        )
    return (
        f"{prefix}Unnecessary Parallelism: A hardcoded thread count "
        "greater than 1 was detected. Setting a fixed number of threads "
        "without knowing the target machine's CPU core count can cause "
        "resource contention and slow down TensorFlow training."
    )


def detect_smell_in_content(content, file_path):
    """
    Scan file content for Unnecessary Parallelism smell.
    Detection rule: any hardcoded thread value greater than 1.
    Separates findings into two categories:
    1. Active smells - lines that actually execute
    2. Commented smells - lines that are commented out
    """
    active_findings = []
    commented_findings = []

    try:
        decoded = base64.b64decode(content).decode("utf-8", errors="ignore")

        # Only process files that use TensorFlow or Keras
        if not is_tensorflow_file(decoded):
            return [], []

        lines = decoded.split("\n")
        for line_num, line in enumerate(lines, start=1):

            # Check if line contains the smell
            if not is_smell(line):
                continue

            if is_commented_line(line):
                # Line is commented out - bad code hygiene
                commented_findings.append({
                    "line_number": line_num,
                    "code_snippet": line.strip(),
                    "explanation": generate_explanation(
                        line.strip(),
                        is_commented=True
                    )
                })
            else:
                # Line is active code - real smell
                active_findings.append({
                    "line_number": line_num,
                    "code_snippet": line.strip(),
                    "explanation": generate_explanation(
                        line.strip(),
                        is_commented=False
                    )
                })

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

    return active_findings, commented_findings


def main():
    print("Starting Unnecessary Parallelism Detector...")
    print("=" * 60)
    print("Detection Rule: Hardcoded thread value > 1 is flagged")
    print("Setting to 1 means no parallelism - different problem")
    print("Active smells    → unnecessary_parallelism_results.csv")
    print("Commented smells → unnecessary_parallelism_commented_results.csv")
    print("=" * 60)

    check_rate_limit()

    # Load existing results to resume if interrupted
    active_results, commented_results, seen_active, seen_commented = load_existing_results()
    seen_files = set()
    total_scanned = 0

    for query in SEARCH_QUERIES:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")

        search_remaining, _ = check_rate_limit()
        wait_if_needed(search_remaining)

        page = 1
        while page <= 10:
            print(f"  Fetching page {page}...")
            items, total_count = search_code(query, page)

            if not items:
                break

            print(f"  Total on GitHub: {total_count} | This page: {len(items)}")

            for item in items:
                file_path = item.get("path", "")
                repo_name = item["repository"]["full_name"]
                repo_url = item["repository"]["html_url"]
                file_url = item.get("url", "")

                # Skip already scanned files
                unique_key = f"{repo_name}/{file_path}"
                if unique_key in seen_files:
                    continue
                seen_files.add(unique_key)
                total_scanned += 1

                # Apply repository quality filters
                repo_details = get_repo_details(repo_name)
                if not passes_filters(repo_details):
                    continue

                print(f"  Scanning: {repo_name} -> {file_path} ⭐{repo_details['stargazers_count']}")

                # Download and analyze file content
                content = get_file_content(file_url)
                if not content:
                    continue

                active_findings, commented_findings = detect_smell_in_content(
                    content, file_path
                )

                # Save active smells to main CSV
                for finding in active_findings:
                    smell_key = (repo_name, file_path, finding["line_number"])
                    if smell_key in seen_active:
                        continue
                    seen_active.add(smell_key)
                    result = {
                        "repository_name": repo_name,
                        "repository_url": repo_url,
                        "file_path": file_path,
                        "line_number": finding["line_number"],
                        "code_snippet": finding["code_snippet"],
                        "explanation": finding["explanation"]
                    }
                    save_active_result(result, active_results)
                    print(f"  Active smell at line {finding['line_number']}: {finding['code_snippet'][:60]}")

                # Save commented smells to secondary CSV
                for finding in commented_findings:
                    smell_key = (repo_name, file_path, finding["line_number"])
                    if smell_key in seen_commented:
                        continue
                    seen_commented.add(smell_key)
                    result = {
                        "repository_name": repo_name,
                        "repository_url": repo_url,
                        "file_path": file_path,
                        "line_number": finding["line_number"],
                        "code_snippet": finding["code_snippet"],
                        "explanation": finding["explanation"]
                    }
                    save_commented_result(result, commented_results)
                    print(f"  Commented smell at line {finding['line_number']}: {finding['code_snippet'][:60]}")

            if len(items) < 100:
                break

            page += 1
            time.sleep(5)

        print(f"\nTotal files scanned: {total_scanned}")
        print(f"Active smells found: {len(active_results)}")
        print(f"Commented smells found: {len(commented_results)}")

    print(f"\n{'=' * 60}")
    print(f"Done! Scanned {total_scanned} files.")
    print(f"Active smells: {len(active_results)} → {ACTIVE_CSV}")
    print(f"Commented smells: {len(commented_results)} → {COMMENTED_CSV}")


if __name__ == "__main__":
    main()
