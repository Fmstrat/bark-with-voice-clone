#!/usr/bin/env bash

set -e

if [ ! -d ./cache ]; then
    mkdir ./cache
fi

docker run \
    -ti \
    --rm \
    --gpus all \
    --name bark \
    -v "${PWD}:${PWD}" \
    -v "${PWD}/cache:/home/user/.cache" \
    -w "${PWD}" \
    --user "$(id -u):$(id -g)" \
    bark $@

# /home/user/.cache/serp/bark_v0/text_2.pt
# -e "SERP_USE_SMALL_MODELS=True" \
