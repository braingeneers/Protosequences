build:
    docker build -t atspaeth/organoid-hmm .  

debug: build
    docker run --rm -it atspaeth/organoid-hmm bash

push: build
    docker push atspaeth/organoid-hmm

queue src exp bin_size ks surrogate="real" method="default":
    #! /usr/bin/env bash
    python queue_hmms.py "{{src}}" "{{exp}}" "{{bin_size}}" "{{ks}}" "{{surrogate}}" "{{method}}"

add-worker n="1":
    #! /usr/bin/env bash
    for i in $(seq "{{n}}"); do
        stamp=$(printf '%(%m%d%H%M%S)T\n' -1)
        export JOB_NAME=atspaeth-hmm-worker-$stamp
        envsubst < hmm_worker.yml | kubectl apply -f -
    done
