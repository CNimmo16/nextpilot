from dotenv import load_dotenv
load_dotenv()

from src.preprocessing import search_repos, download_and_save_repo  # noqa: E402


print("Searching repos...")
repos = search_repos()
print(f"Found {len(repos)} repositories")

for idx, repo in enumerate(repos):
    print(f"Processing {idx+1}/{len(repos)}: {repo['full_name']}")
    download_and_save_repo(repo)

print("Done")
