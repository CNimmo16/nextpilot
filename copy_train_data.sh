#!/bin/bash

copy_data_to_remote () {
    read -p "Enter an ip or host to copy data to: " ip
    read -p "Enter a port: " port

    username=ubuntu

    remote_homedir="$(ssh ${username}@${ip} -p ${port} "pwd")"

    repo_name=distillation

    echo "copying files"

    ssh ${username}@${ip} -p ${port} "mkdir -p ${remote_homedir}/$repo_name/data/nextjs_repos"

    scp -P ${port} -r ./data/nextjs_repos ${username}@${ip}:${remote_homedir}/$repo_name/data

    echo ">> done"
}

copy_data_to_remote
