build:
    docker build -t atspaeth/organoid-hmm .  

debug: build
    docker run --rm -it atspaeth/organoid-hmm bash

push: build
    docker push atspaeth/organoid-hmm

deploy src exp bin_size ks method="default":
    #! /usr/bin/env bash
    export JOB_NAME={{src}}-$(echo {{exp}} | tr _ -)-{{bin_size}}-{{ks}}-{{method}}
    export HMM_DATA_SOURCE={{src}}
    export HMM_EXPERIMENT={{exp}}
    export HMM_BIN_SIZE_MS={{bin_size}}
    export HMM_K_RANGE={{ks}}
    export HMM_METHOD={{method}}
    envsubst < job.yml | kubectl apply -f -

local src exp bin_size ks method="default":
    python stash_hmms.py {{src}} {{exp}} {{bin_size}} {{ks}} {{method}}
