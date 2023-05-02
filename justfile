build:
    docker build -t atspaeth/organoid-hmm .  

debug: build
    docker run --rm -it atspaeth/organoid-hmm bash

push: build
    docker push atspaeth/organoid-hmm

deploy src exp bin_size ks surrogate="real" method="default":
    #! /usr/bin/env bash

    export HMM_DATA_SOURCE="{{src}}"
    export HMM_EXPERIMENT="{{exp}}"
    export HMM_BIN_SIZE_MS="{{bin_size}}"
    export HMM_K_RANGE="{{ks}}"
    export HMM_SURROGATE="{{surrogate}}"
    export HMM_METHOD="{{method}}"

    if [ "$HMM_EXPERIMENT" = "*" ]; then
        s3dir=s3://braingeneers/personal/atspaeth/data/{{src}}/
        s3files=$(aws s3 ls "$s3dir" | grep '[^ ]*\.mat' -o)
        for file in $s3files; do
            exp=$(basename "$file" .mat)
            just deploy {{src}} $exp {{bin_size}} {{ks}} {{surrogate}} {{method}}
        done

    else
        function cleanup_name() {
            echo "$1" | tr '_[:upper:]\,' '-[:lower:].'
        }
        exp=$(cleanup_name $HMM_EXPERIMENT)
        surr=$(cleanup_name $HMM_SURROGATE)
        src=$(cleanup_name $HMM_DATA_SOURCE)
        stamp=$(printf '%(%m%d%H%M%S)T\n' -1)
        export JOB_NAME=atspaeth-hmms-$stamp-$src
        envsubst < job.yml | kubectl apply -f - || exit 1
        kubectl label job $JOB_NAME user=atspaeth app=organoid-hmm \
            data-source=$src surrogate=$surr \
            hmm-method=$HMM_METHOD experiment=$exp
    fi

local src exp bin_size ks surrogate="real" method="default":
    python stash_hmms.py {{src}} {{exp}} {{bin_size}} {{ks}} {{surrogate}} {{method}}
