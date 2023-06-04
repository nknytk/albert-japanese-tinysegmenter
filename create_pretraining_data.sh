#!/bin/bash

. .venv/bin/activate

mkdir -p logs

function create_data() {
  tokenizer=$1
  n=$2

  t_dir=${tokenizer}
  if [ "${tokenizer}" == "wordpiece" ]; then
    t_dir=wordpiece_tinysegmenter
  fi
  mkdir -p data/pretrain/${t_dir}
  python -u create_pretraining_data.py \
    --input_file data/corpus/corpus_${n}.txt \
    --output_file data/pretrain/${t_dir}/pretrain_${n}.tfrecord.gz \
    --vocab_file models/tokenizers/${t_dir}/vocab.txt \
    --tokenizer_type ${tokenizer} > logs/${tokenizer}_${n}.log
}

export -f create_data

# Run parallel 5 data creation process.
# 2.5GB memory, 14~15 minutes per process on Ryzen 1700.
seq 0 639 | xargs -I % -P 5 bash -c "create_data wordpiece %"
 
# Run parallel 4 data creation process.
# 3.1GB memory, 16~17 minutes per process on Ryzen 1700.
seq 0 639 | xargs -I % -P 4 bash -c "create_data character %"
