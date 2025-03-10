#!/bin/bash

copy_data_to_remote () {
    read -p "Enter an ip or host to copy data to: " ip
    read -p "Enter a port: " port

    username=ubuntu

    remote_homedir="$(ssh ${username}@${ip} -p ${port} "pwd")"

    repo_name=distillation

    echo "zipping..."
    
    zip -r data/nextjs_repos.zip data/nextjs_repos

    echo "copying files"

    ssh ${username}@${ip} -p ${port} "mkdir -p ${remote_homedir}/$repo_name/data"

    scp -P ${port} -r ./data/nextjs_repos.zip ${username}@${ip}:${remote_homedir}/$repo_name/data

    ssh ${username}@${ip} -p ${port} "rm -rf ${remote_homedir}/$repo_name/data/nextjs_repos"
    ssh ${username}@${ip} -p ${port} "unzip ${remote_homedir}/$repo_name/data/nextjs_repos.zip -d ${remote_homedir}/$repo_name/data/nextjs_repos"

    echo ">> done"
}

copy_data_to_remote
