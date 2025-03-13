#!/bin/bash

open_remote () {
    read -p "Enter an ip or host to connect to: " ip
    read -p "Enter a port: " port
    read -p "Enter a username: " username
    
    repo_ref="cnimmo16/nextpilot"

    repo_url="https://github.com/$repo_ref"

    repo_name=$(basename "$repo_url")

    remote_homedir="$(ssh ${username}@${ip} -p ${port} "pwd")"

    echo "cloning repo"

    ssh ${username}@${ip} -p ${port} -t "git clone ${repo_url}"

    echo "copying .env file"

    scp -P ${port} ./.env ${username}@${ip}:${remote_homedir}/${repo_name}/.env

    echo "remote_homedir: ${remote_homedir}"

    repo_path="$remote_homedir/$repo_name"

    echo "repo path: ${repo_path}"
    
    echo "opening ${repo_path} on remote"

    code --remote ssh-remote+${username}@${ip}:${port} $repo_path
}

open_remote
