import os
import requests
import time
import urllib
import base64

# Configuration
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
OUTPUT_DIR = "data/nextjs_repos"
PER_PAGE = 50  # Results per page
MAX_REPOS = 200  # Max repositories to process
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
    
    if remaining <= 10:  # Buffer to avoid hitting the limit
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
        'topic:nextjs topic:template '
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

def has_next_config(repo_full_name):
    """Check if repository has a next.config file with any valid extension"""
    valid_extensions = ['js', 'ts', 'mjs']
    
    for ext in valid_extensions:
        config_url = f"https://api.github.com/repos/{repo_full_name}/contents/next.config.{ext}"
        try:
            response = make_github_request(config_url)
            return True
        except Exception as e:
            continue

    return False

def has_app_directory(repo_full_name):
    """Check if repository contains an /app directory within 3 levels"""
    queue = [{'path': '', 'depth': 0}]
    
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

def get_ts_files(repo_full_name):
    """Get all .ts and .tsx files from a repository"""
    files = []
    page = 1
    
    while True:
        url = (
            f"https://api.github.com/search/code?"
            f"q=repo:{repo_full_name}+extension:ts+extension:tsx&per_page=100&page={page}"
        )
        response = make_github_request(url)
        
        if response.status_code != 200:
            print(f"Error searching code: {response.json()}")
            break
            
        data = response.json()
        files.extend(data['items'])
        
        if len(data['items']) < 100:
            break
            
        page += 1
        
    return files

def download_and_save_repo(repo):
    """Process a repository and save its TypeScript files"""
    repo_name = repo['name']
    owner = repo['owner']['login']
    repo_full_name = f"{owner}/{repo_name}"
    
    # Check for next.config.js
    if not has_next_config(repo_full_name):
        print('no next config, skipping')
        return
        
    # Check for app directory
    if not has_app_directory(repo_full_name):
        print('no app dir, skipping')
        return
        
    # Get all TS files
    files = get_ts_files(repo_full_name)
    
    # Save to file
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    chunk_idx = 0
    content_len = 0
    for file in files:
        output_path = os.path.join(OUTPUT_DIR, f"{repo_name.replace('/', '_')}_{chunk_idx}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
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
