import os
import re
import time
import requests
import pandas as pd
from dotenv import load_dotenv

# Load token from .env file
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Headers for GitHub API
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Multiple search queries to maximize coverage
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

# Detection patterns
SMELL_PATTERNS = [
    r'os\.environ\[.LD_LIBRARY_PATH.\]\s*=',
    r'os\.environ\[.CUDA_VISIBLE_DEVICES.\]\s*=',
    r'os\.environ\.get\(.LD_LIBRARY_PATH',
    r'os\.environ\.get\(.CUDA_VISIBLE_DEVICES',
]

# TensorFlow import patterns
TENSORFLOW_PATTERNS = [
    r'import tensorflow',
    r'import tensorflow as tf',
    r'from tensorflow',
    r'import keras',
    r'from keras',
    r'from tf\.keras',
]

# CSV file to save results
CSV_FILE = "library_path_mismatch_results.csv"

def load_existing_results():
    """Load already saved results to avoid duplicates"""
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        print(f"📂 Loaded {len(df)} existing results from CSV")
        seen = set(zip(df["repository_name"], df["file_path"], df["line_number"]))
        return df.to_dict("records"), seen
    return [], set()

def save_result(result, results):
    """Save a single result to CSV immediately"""
    results.append(result)
    df = pd.DataFrame(results)
    df.drop_duplicates(subset=["repository_name", "file_path", "line_number"], inplace=True)
    df.to_csv(CSV_FILE, index=False)

def check_rate_limit():
    """Check remaining API requests"""
    try:
        response = requests.get("https://api.github.com/rate_limit", headers=HEADERS, timeout=10)
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
    """Wait if rate limit is low"""
    if remaining < threshold:
        print(f"Rate limit low! Waiting 60 seconds...")
        time.sleep(60)

def safe_get(url, params=None, retries=3):
    """Make API request with retry logic"""
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=15)
            return response
        except Exception as e:
            print(f"Connection error (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                print(f"Waiting 10 seconds before retry...")
                time.sleep(10)
    return None

def search_code(query, page=1):
    """Search GitHub code directly with pagination"""
    url = "https://api.github.com/search/code"
    params = {
        "q": query,
        "per_page": 100,
        "page": page
    }
    response = safe_get(url, params=params)
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
    else:
        print(f"Search failed: {response.status_code}")
        return [], 0

def get_repo_details(repo_full_name):
    """Get repository details to apply filters"""
    url = f"https://api.github.com/repos/{repo_full_name}"
    response = safe_get(url)
    time.sleep(1)
    if response and response.status_code == 200:
        return response.json()
    return None

def get_file_content(url):
    """Download and return file content"""
    response = safe_get(url)
    time.sleep(1)
    if response and response.status_code == 200:
        return response.json().get("content", "")
    return ""

def is_tensorflow_file(decoded_content):
    """Check if file uses TensorFlow or Keras"""
    for pattern in TENSORFLOW_PATTERNS:
        if re.search(pattern, decoded_content, re.IGNORECASE):
            return True
    return False

def detect_smell_in_content(content, file_path):
    """Detect library path mismatch smell in file content"""
    findings = []
    try:
        import base64
        decoded = base64.b64decode(content).decode("utf-8", errors="ignore")
        if not is_tensorflow_file(decoded):
            return []
        lines = decoded.split("\n")
        for line_num, line in enumerate(lines, start=1):
            for pattern in SMELL_PATTERNS:
                if re.search(pattern, line):
                    findings.append({
                        "line_number": line_num,
                        "code_snippet": line.strip(),
                        "explanation": generate_explanation(line.strip())
                    })
    except Exception as e:
        print(f"Error decoding {file_path}: {e}")
    return findings

def generate_explanation(snippet):
    """Generate explanation based on detected pattern"""
    if "LD_LIBRARY_PATH" in snippet:
        return (
            "Library Path Mismatch: LD_LIBRARY_PATH is hardcoded. "
            "This can point to non-existent or wrong paths causing "
            "TensorFlow to slow down or fail when loading libraries."
        )
    elif "CUDA_VISIBLE_DEVICES" in snippet:
        return (
            "Library Path Mismatch: CUDA_VISIBLE_DEVICES is hardcoded. "
            "This forces a specific GPU device which may not exist on "
            "other machines breaking portability."
        )
    return "Library Path Mismatch detected: hardcoded environment path."

def passes_filters(repo_details):
    """Apply repository filters"""
    if not repo_details:
        return False
    return (
        repo_details.get("stargazers_count", 0) >= 10 and
        not repo_details.get("archived", True) and
        not repo_details.get("disabled", True) and
        repo_details.get("language") == "Python"
    )

def main():
    print("Starting Library Path Mismatch Detector...")
    print("=" * 60)

    search_remaining, core_remaining = check_rate_limit()

    # Load existing results to avoid losing data on crash
    results, seen_smells = load_existing_results()
    seen_files = set()

    total_scanned = 0

    for query in SEARCH_QUERIES:
        print(f"\n{'='*60}")
        print(f"Query: {query}")

        search_remaining, core_remaining = check_rate_limit()
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

                unique_key = f"{repo_name}/{file_path}"
                if unique_key in seen_files:
                    continue
                seen_files.add(unique_key)
                total_scanned += 1

                repo_details = get_repo_details(repo_name)
                if not passes_filters(repo_details):
                    continue

                print(f"  ✅ Scanning: {repo_name} -> {file_path} ⭐{repo_details['stargazers_count']}")

                content = get_file_content(file_url)
                if not content:
                    continue

                findings = detect_smell_in_content(content, file_path)

                for finding in findings:
                    smell_key = (repo_name, file_path, finding["line_number"])
                    if smell_key in seen_smells:
                        continue
                    seen_smells.add(smell_key)

                    result = {
                        "repository_name": repo_name,
                        "repository_url": repo_url,
                        "file_path": file_path,
                        "line_number": finding["line_number"],
                        "code_snippet": finding["code_snippet"],
                        "explanation": finding["explanation"]
                    }

                    # Save immediately after each finding
                    save_result(result, results)
                    print(f"  🎯 Smell found at line {finding['line_number']}: {finding['code_snippet'][:60]}")

            if len(items) < 100:
                break

            page += 1
            time.sleep(5)

        print(f"\nTotal files scanned: {total_scanned}")
        print(f"Total smells found: {len(results)}")

    print(f"\n{'='*60}")
    print(f"✅ Done! Scanned {total_scanned} files.")
    print(f"✅ Found {len(results)} unique smells.")
    print(f"📄 Results saved to {CSV_FILE}")

if __name__ == "__main__":
    main()
