#!/bin/bash
# 
#  2024   jerrypeng1937@gmail.com
#  demo on the usage

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

umask 000


stage=0.0
is_demo=true
. utils/parse_options.sh


if (( $(echo "$stage <= 0.0" | bc -l) )); then
    python code/preprocess/clean.py \
        data/demo \
        data/clean || { log "failed to clean"; exit 1; }
fi

if (( $(echo "$stage <= 0.1" | bc -l) )); then
    python code/preprocess/chunk.py run \
        data/clean \
        data/chunk.jsonl || { log "failed to chunk"; exit 1; }
fi

if (( $(echo "$stage <= 0.2" | bc -l) )); then
    python code/parallel_request.py \
        --model text-embedding-3-small \
        data/chunk.jsonl \
        data/embed.jsonl || { log "failed to do parallel_request"; exit 1; }
fi

if (( $(echo "$stage <= 0.3" | bc -l) )); then
    python code/vectordb.py \
        data/chunk.jsonl \
        data/embed.jsonl || { log "failed to create vectordb"; exit 1; }
fi

if (( $(echo "$stage <= 1.0" | bc -l) )); then
    if [ "$is_demo" = true ]; then
        python code/main.py demo \
            data/chunk.jsonl \
            data/embed.jsonl
    else
        python code/main.py batch \
            data/question.jsonl \
            data/answer.jsonl
    fi
fi
