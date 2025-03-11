import os
import requests
import time
import urllib
import base64

# Configuration
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
OUTPUT_DIR = "data/nextjs_repos"
PER_PAGE = 50  # Results per page
MAX_REPOS = 1000  # Max repositories to process
MAX_LENGTH_PER_FILE = 5_000

headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

def check_rate_limit(response):
    """Check rate limit headers and sleep if necessary."""
    if 'X-RateLimit-Remaining' not in response.headers:
        return
    
    remaining = int(response.headers['X-RateLimit-Remaining'])
    reset_time = int(response.headers['X-RateLimit-Reset'])  # Unix timestamp
    
    if remaining <= 5:  # Buffer to avoid hitting the limit
        sleep_duration = max(reset_time - time.time(), 0) + 10  # Add 10 seconds buffer
        print(f"Rate limit approaching. Sleeping for {sleep_duration:.1f} seconds...")
        time.sleep(sleep_duration)

def make_github_request(url):
    """Make a GitHub API request with rate limit handling."""
    while True:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            check_rate_limit(response)
            return response
        elif response.status_code == 403 and 'rate limit' in response.text.lower():
            # Handle rate limit exceeded
            reset_time = int(response.headers['X-RateLimit-Reset'])
            sleep_duration = max(reset_time - time.time(), 0) + 10  # Add 10 seconds buffer
            print(f"Rate limit exceeded. Sleeping for {sleep_duration:.1f} seconds...")
            time.sleep(sleep_duration)
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")

def search_repos():
    """Search for TypeScript repos with Next.js topics created after 2024-01-01"""
    base_query = (
        'language:TypeScript '
        'topic:nextjs '
        'created:>=2024-01-01'
    )
    repos = []
    page = 1
    
    while len(repos) < MAX_REPOS:
        url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(base_query)}&sort=stars&order=desc&per_page={PER_PAGE}&page={page}"
        response = make_github_request(url)
        
        if response.status_code != 200:
            print(f"Error searching repos: {response.json()}")
            break
            
        data = response.json()
        repos.extend(data['items'])
        
        if len(data['items']) < PER_PAGE:
            break
            
        page += 1
        
    return repos[:MAX_REPOS]

def get_next_directories(repo_full_name):
    """Check if repository has a next.config file with any valid extension within 3 levels"""
    queue = [{'path': '', 'depth': 0}]

    next_dirs = []

    while queue:
        current = queue.pop(0)
        if current['depth'] > 3:
            continue
            
        url = f"https://api.github.com/repos/{repo_full_name}/contents/{current['path']}"
        response = make_github_request(url)
        
        if response.status_code == 200:
            contents = response.json()
            for item in contents:
                if item['type'] == 'dir':
                    queue.append({
                        'path': f"{current['path']}/{item['name']}".lstrip('/'),
                        'depth': current['depth'] + 1
                    })
                elif item['name'].startswith('next.config.'):
                    next_dirs.append(current['path'])
        elif response.status_code != 404:
            print(f"Error checking directory: {response.json()}")

    return next_dirs

def has_app_directory(repo_full_name, next_dir):
    """Check if repository contains an /app directory within 3 levels"""
    queue = [{'path': next_dir, 'depth': 0}]
    
    while queue:
        current = queue.pop(0)
        if current['depth'] > 3:
            continue
            
        url = f"https://api.github.com/repos/{repo_full_name}/contents/{current['path']}"
        response = make_github_request(url)
        
        if response.status_code == 200:
            contents = response.json()
            for item in contents:
                if item['type'] == 'dir' and item['name'] == 'app':
                    return True
                elif item['type'] == 'dir':
                    queue.append({
                        'path': f"{current['path']}/{item['name']}".lstrip('/'),
                        'depth': current['depth'] + 1
                    })
        elif response.status_code != 404:
            print(f"Error checking directory: {response.json()}")
        
    return False

def get_ts_files(repo_full_name, next_dir):
    """Get all .ts and .tsx files from a repository"""
    url = f"https://api.github.com/repos/{repo_full_name}/contents/{next_dir}"
    ts_files = []

    def fetch_files(_url: str):
        response = requests.get(_url, headers=headers)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as err:
            print(f"Error fetching file at: {_url} - {err}")
            return
        items = response.json()
        for item in items:
            if item["type"] == "file" and (item["name"].endswith(".ts") or item["name"].endswith(".tsx")):
                ts_files.append(item)
            elif item["type"] == "dir":
                fetch_files(item["url"])

    fetch_files(url)
    return ts_files

def download_and_save_repo(repo):
    """Process a repository and save its TypeScript files"""
    repo_name = repo['name']
    owner = repo['owner']['login']
    repo_full_name = f"{owner}/{repo_name}"
    
    # Check for next.config.js
    next_dirs = get_next_directories(repo_full_name)
    if len(next_dirs) == 0:
        print('no next configs, skipping')
        return
    
    for next_dir in next_dirs:            
        # Check for app directory
        if not has_app_directory(repo_full_name, next_dir):
            print(f"no app dir at /{next_dir}, skipping")
            return
            
        # Get all TS files
        files = get_ts_files(repo_full_name, next_dir)
        
        # Save to file
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        chunk_idx = 0
        content_len = 0
        for file in files:
            namespaced_loc = f"{repo_full_name}/{next_dir}"
            output_path = os.path.join(OUTPUT_DIR, f"{namespaced_loc.replace('/', '__')}_{chunk_idx}.txt")
            with open(output_path, 'a', encoding='utf-8') as f:
                try:
                    content_url = file['url']
                    content = make_github_request(content_url).json()['content']
                    content_len += len(content)
                    # Decode base64 content
                    f.write(f"// File: {file['path']}\n")
                    f.write(f"{base64.b64decode(content).decode('utf-8')}\n\n")
                    if content_len > MAX_LENGTH_PER_FILE:
                        content_len = 0
                        chunk_idx += 1
                except Exception as e:
                    print(f"Error downloading {file['path']}: {str(e)}")
