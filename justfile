export container := "atspaeth/organoid-hmm"

list:
    @just --list

build:
    docker build -t $container .

debug: pull
    docker run --rm -it $container bash

pull:
    docker pull $container

push: build
    docker push $container

queue src exp bin_size ks surrogate="real" method="default":
    python fit_hmms.py "{{src}}" "{{exp}}" "{{bin_size}}" "{{ks}}" "{{surrogate}}" "{{method}}"

add-worker n="1":
    #! /usr/bin/bash
    if [ -z "$S3_USER" ]; then
        echo \$S3_USER must be defined. >&2
        exit 1
    fi
    : ${PRP_MEMORY_GI:=4}
    : ${PRP_MEMORY_LIMIT_GI:=$(( ${PRP_MEMORY_GI} * 14 / 10 ))}
    export PRP_MEMORY_GI
    export PRP_MEMORY_LIMIT_GI
    for i in $(seq "{{n}}"); do
        stamp=$(printf '%(%m%d%H%M%S)T\n' -1)
        export JOB_NAME=atspaeth-hmm-worker--$stamp$i
        envsubst < hmm_worker.yml | kubectl apply -f -
    done
