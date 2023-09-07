export container := "atspaeth/organoid-hmm"

help:
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

add-worker for="hmm" n="1" memory_gi="4":
    #! /usr/bin/bash
    if [ -z "$S3_USER" ]; then
        echo \$S3_USER must be defined. >&2
        exit 1
    fi
    export WORKER_TYPE={{for}}
    export CONTAINER_IMAGE={{container}}
    export NRP_MEMORY_GI={{memory_gi}}
    export NRP_MEMORY_LIMIT_GI=$(( {{memory_gi}} * 14 / 10 ))
    echo "Running with {{memory_gi}}GiB RAM"
    for i in $(seq "{{n}}"); do
        stamp=$(printf '%(%m%d%H%M%S)T\n' -1)
        export JOB_NAME=$S3_USER-{{for}}-worker--$stamp$i
        envsubst < worker.yml | kubectl apply -f -
    done
