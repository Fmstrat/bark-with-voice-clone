#!/usr/bin/env bash

set -e

if [ ! -d ./cache ]; then
    mkdir ./cache
fi
if [ ! -d ./voices ]; then
    mkdir ./voices
fi

CMD="python3 generate.py"
MNT=""
if [ "${1}" = "clone_generate" ]; then
	MNT=" -v ${PWD}/voices:${PWD}/bark/assets/prompts"
	CMD="python3 clone_generate.py"
fi
if [ "${1}" = "clone_voice" ]; then
	CMD="python3 clone_voice.py"
fi
shift;

set -x
docker run \
    -ti \
    --rm \
    --gpus all \
    --name bark \
    -v "${PWD}:${PWD}" \
    -v "${PWD}/cache:/home/user/.cache" \
    ${MNT} \
    -w "${PWD}" \
    --user "$(id -u):$(id -g)" \
    bark ${CMD} $@

# /home/user/.cache/serp/bark_v0/text_2.pt
# -e "SERP_USE_SMALL_MODELS=True" \
