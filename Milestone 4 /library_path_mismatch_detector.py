
import os
import re
import time
import requests
import pandas as pd
from dotenv import load_dotenv

# Load token from .env file
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Multiple search queries to maximize coverage
# Each query targets a different variation of the smell
# We use both tensorflow and keras to cover all cases
SEARCH_QUERIES = [
    "os.environ LD_LIBRARY_PATH tensorflow language:python",
    "os.environ CUDA_VISIBLE_DEVICES tensorflow language:python",
    "LD_LIBRARY_PATH tensorflow language:python",
    "CUDA_VISIBLE_DEVICES tensorflow language:python",
    "os.environ LD_LIBRARY_PATH keras language:python",
    "os.environ CUDA_VISIBLE_DEVICES keras language:python",
    "LD_LIBRARY_PATH import tensorflow language:python",
    "CUDA_VISIBLE_DEVICES import tensorflow language:python",
    "os.environ LD_LIBRARY_PATH tf.keras language:python",
    "os.environ CUDA_VISIBLE_DEVICES tf.keras language:python",
]

# Detection patterns for active smell lines
SMELL_PATTERNS = [
    r'os\.environ\[.LD_LIBRARY_PATH.\]\s*=',
    r'os\.environ\[.CUDA_VISIBLE_DEVICES.\]\s*=',
    r'os\.environ\.get\(.LD_LIBRARY_PATH',
    r'os\.environ\.get\(.CUDA_VISIBLE_DEVICES',
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
# 1. Active smells - lines that actually execute (our main results)
# 2. Commented smells - lines commented out (bad code hygiene evidence)
ACTIVE_CSV = "library_path_mismatch_results.csv"
COMMENTED_CSV = "library_path_mismatch_commented_results.csv"


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
    Active smells are lines that actually execute in the code.
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
    Save a commented smell result immediately to the secondary CSV.
    Commented smells are lines starting with # that do not execute.
    These serve as evidence of bad code hygiene in the codebase.
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
    This keeps our results consistent with M1-M3 framework
    and ensures we only flag relevant ML code.
    """
    for pattern in TENSORFLOW_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return True
    return False


def is_commented_line(line):
    """
    Check if a line is commented out.
    A commented line starts with # (ignoring whitespace).
    Commented smells do not execute but still represent
    bad code hygiene and are saved separately.
    """
    return line.strip().startswith("#")


def generate_explanation(snippet, is_commented=False):
    """Generate explanation based on the detected pattern."""
    prefix = "COMMENTED - Bad Code Hygiene: " if is_commented else ""

    if "LD_LIBRARY_PATH" in snippet:
        return (
            f"{prefix}Library Path Mismatch: LD_LIBRARY_PATH is hardcoded. "
            "This can point to non-existent or wrong paths causing "
            "TensorFlow to slow down or fail when loading libraries."
        )
    elif "CUDA_VISIBLE_DEVICES" in snippet:
        return (
            f"{prefix}Library Path Mismatch: CUDA_VISIBLE_DEVICES is hardcoded. "
            "This forces a specific GPU device which may not exist on "
            "other machines breaking portability."
        )
    return f"{prefix}Library Path Mismatch: hardcoded environment path detected."


def detect_smell_in_content(content, file_path):
    """
    Scan file content for Library Path Mismatch smell.
    Separates findings into two categories:
    1. Active smells - lines that actually execute
    2. Commented smells - lines that are commented out
    This separation keeps our research accurate while
    still capturing evidence of bad code hygiene.
    """
    active_findings = []
    commented_findings = []

    try:
        import base64
        decoded = base64.b64decode(content).decode("utf-8", errors="ignore")

        # Only process files that use TensorFlow or Keras
        if not is_tensorflow_file(decoded):
            return [], []

        lines = decoded.split("\n")
        for line_num, line in enumerate(lines, start=1):
            for pattern in SMELL_PATTERNS:
                if re.search(pattern, line):
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
                    break

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

    return active_findings, commented_findings


def main():
    print("Starting Library Path Mismatch Detector...")
    print("=" * 60)
    print("Active smells   → library_path_mismatch_results.csv")
    print("Commented smells → library_path_mismatch_commented_results.csv")
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
