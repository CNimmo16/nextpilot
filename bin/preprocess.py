import preprocessing

repos = preprocessing.search_repos()
print(f"Found {len(repos)} repositories")

for idx, repo in enumerate(repos):
    print(f"Processing {idx+1}/{len(repos)}: {repo['full_name']}")
    preprocessing.download_and_save_repo(repo)

print("Done")
