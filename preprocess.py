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
OFFSET = 0

headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

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
        url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(base_query)}&sort=stars&order=desc&per_page={PER_PAGE}&page={page}&offset={OFFSET}"
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"Error searching repos: {response.json()}")
            break
            
        data = response.json()
        repos.extend(data['items'])
        
        if len(data['items']) < PER_PAGE:
            break
            
        page += 1
        time.sleep(2)  # Rate limit handling
        
    return repos[:MAX_REPOS]

def has_app_directory(repo_full_name):
    """Check if repository contains an /app directory within 3 levels"""
    queue = [{'path': '', 'depth': 0}]
    
    while queue:
        current = queue.pop(0)
        if current['depth'] > 3:
            continue
            
        url = f"https://api.github.com/repos/{repo_full_name}/contents/{current['path']}"
        response = requests.get(url, headers=headers)
        
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
            
        time.sleep(0.5)
        
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
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"Error searching code: {response.json()}")
            break
            
        data = response.json()
        files.extend(data['items'])
        
        if len(data['items']) < 100:
            break
            
        page += 1
        time.sleep(2)
        
    return files

def download_and_save_repo(repo):
    """Process a repository and save its TypeScript files"""
    repo_name = repo['name']
    owner = repo['owner']['login']
    repo_full_name = f"{owner}/{repo_name}"
    
    # Check for next.config.js
    config_url = f"https://api.github.com/repos/{repo_full_name}/contents/next.config.js"
    if requests.get(config_url, headers=headers).status_code != 200:
        return
        
    # Check for app directory
    if not has_app_directory(repo_full_name):
        print('no app dir, skipping')
        return
        
    # Get all TS files
    files = get_ts_files(repo_full_name)
    
    # Save to file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{repo_name.replace('/', '_')}.txt")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for file in files:
            try:
                content_url = file['url']
                content = requests.get(content_url, headers=headers).json()['content']
                # Decode base64 content
                f.write(f"// File: {file['path']}\n")
                f.write(f"{base64.b64decode(content).decode('utf-8')}\n\n")
                time.sleep(0.5)
            except Exception as e:
                print(f"Error downloading {file['path']}: {str(e)}")

if __name__ == "__main__":
    repos = search_repos()
    print(f"Found {len(repos)} repositories")
    
    for idx, repo in enumerate(repos):
        print(f"Processing {idx+1}/{len(repos)}: {repo['full_name']}")
        download_and_save_repo(repo)
        time.sleep(2)  # Rate limit handling