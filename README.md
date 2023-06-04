# Pretrained Japanese ALBERT Models

This repository consists of pretrained japanese ALBERT models, codes and guides to pretrain these models. These models are planned to also be available at Hugging Face Model Hub.  
A dictionary-free compact japanese tokenizer [TinySegmenter](http://chasen.org/~taku/software/TinySegmenter/) is used for tokenization in these models to take advantage of ALBERT's small memory and disk footprint.

## Models

### Wordpiece models

| model path | num of hidden layers | hidden layer dimensions | intermediate layer dimensions | description |
| --- | --- | --- | --- | --- |
| [models/wordpiece_tinysegmenter/base](./models/wordpiece_tinysegmenter/base) | 12 | 768 | 3072 | same config as original ALBERT-base's |
| [models/wordpiece_tinysegmenter/medium](./models/wordpiece_tinysegmenter/medium) | 9 | 576 |  2304 | my personal recommendation |
| [models/wordpiece_tinysegmenter/small](./models/wordpiece_tinysegmenter/small) | 6 | 384 | 1536 | |
| [models/wordpiece_tinysegmenter/tiny](./models/wordpiece_tinysegmenter/tiny) | 4 | 312 | 1248 | |

### Character models

| model path | num of hidden layers | hidden layer dimensions | intermediate layer dimensions | description |
| --- | --- | --- | --- | --- |
| [models/character/base](./models/character/base) | 12 | 768 | 3072 | same config as original ALBERT-base's |
| [models/character/small](./models/character/small) | 6 | 384 | 1536 | |
| [models/character/tiny](./models/character/tiny) | 4 | 312 | 1248 | |

## Tokenizers

### BertJapaneseTinySegmenterTokenizer

A transformers tokenizer implementation for our japanese ALBERT models.  
This tokenizer use [tinysegmenter3](https://pypi.org/project/tinysegmenter3/) instead of MeCab to take advantage of ALBERT's compactness.  
You can choose `wordpiece` or `character` as `subword_tokenizer_type`. Default subword tokenizer is `wordpiece`.

## Usage Example

### BertJapaneseTinySegmenterTokenizer example

```python
>>> from tokenization import BertJapaneseTinySegmenterTokenizer
>>> vocab_file = 'models/wordpiece_tinysegmenter/base/vocab.txt'
>>> word_tokenizer = BertJapaneseTinySegmenterTokenizer(vocab_file)
>>> word_tokenizer.tokenize('単語単位で分かち書きをします。')
['単語', '単位', 'で', '分か', '##ち', '書き', 'を', 'し', 'ます', '。']
>>> word_tokenizer('単語単位で分かち書きをします。', max_length=16, padding='max_length', truncation=True)
{
    'input_ids': [2, 18968, 14357, 916, 14708, 6287, 13817, 959, 900, 12441, 857, 3, 0, 0, 0, 0],
    'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
}
>>> vocab_file = 'models/character/base/vocab.txt'
>>> char_tokenizer = BertJapaneseTinySegmenterTokenizer(vocab_file, subword_tokenizer_type='character')
>>> char_tokenizer.tokenize('文字単位で分かち書きをします。')
['文', '字', '単', '位', 'で', '分', 'か', 'ち', '書', 'き', 'を', 'し', 'ま', 'す', '。']
>>> char_tokenizer('文字単位で分かち書きをします。', max_length=16, padding='max_length', truncation=True)
{
    'input_ids': [2, 2709, 1979, 1517, 1182, 916, 1402, 888, 910, 2825, 890, 959, 900, 939, 902, 3],
    'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
```

### Albert models example

```python
>>> import torch
>>> from transformers import AlbertForMaskedLM
>>> from tokenization import BertJapaneseTinySegmenterTokenizer
>>> model_size = 'medium'
>>> model_dir = f'models/wordpiece_tinysegmenter/{model_size}'
>>> tokenizer = BertJapaneseTinySegmenterTokenizer(f'{model_dir}/vocab.txt')
>>> model = AlbertForMaskedLM.from_pretrained(model_dir)
>>> text = '個人で[MASK]を研究しています。'
>>> enc = tokenizer(text, max_length=16, padding='max_length', truncation=True)
>>> with torch.no_grad():
...     _input = {k: torch.tensor([v]) for k, v in enc.items()}
...     scores = model(**_input).logits
...     token_ids = scores[0].argmax(-1).tolist()
...
>>> filtered_token_ids = [token_ids[i] for i in range(1, len(token_ids)) if enc['attention_mask']]
>>> tokenizer.convert_ids_to_tokens(filtered_token_ids)
>>> special_ids = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]
>>> filtered_token_ids = [token_ids[i] for i in range(len(token_ids)) if enc['attention_mask'][i] and enc['input_ids'][i] not in special_ids]
>>> tokenizer.convert_ids_to_tokens(filtered_token_ids)
['個人', 'で', '数学', 'を', '研究', 'し', 'て', 'い', 'ます', '。']
```

## Pretraining Guides

### Create training data

Note: The training data used to train models was prepared on a local workstation. The corpus was split into 640 files to fit memory consumption into 16GB. The number of corpus files should be set in accordance with your environment. The number is hardcoded at [make_split_corpus.py#16](./make_split_corpus.py#16) and [create_pretraining_data.sh#27](./create_pretraining_data.sh#27).

Prepare a Python virtualenv.
```bash
$ python3 -m venv .venv
$ . .venv/bin/activate
$ pip install -r requirements.txt
```

The models are trained on japanese Wikipedia.   
Download `jawiki-20230424-cirrussearch-content.json.gz` from https://dumps.wikimedia.org/other/cirrussearch/.

Create split corpus files.
```bash
$ mkdir -p data/corpus
$ python make_split_corpus.py data/jawiki-20230424-cirrussearch-content.json.gz data/corpus
```

Sample corpus to train sub tokenizer.
```bash
$ grep -v '^$' data/corpus/*.txt | shuf | head -3000000 > data/corpus_sampled.txt
```

Train sub tokenizer.
```bash
$ mkdir -p models/tokenizers/wordpiece_tinysegmenter
$ TOKENIZERS_PARALLELISM=false python train_tokenizer.py \
  --input_files data/corpus_sampled.txt \
  --output_dir models/tokenizers/wordpiece_tinysegmenter \
  --tokenizer_type wordpiece \
  --vocab_size 32768 \
  --limit_alphabet 6129 \
  --num_unused_tokens 10
$ mkdir -p models/tokenizers/character
$ head -6144 models/tokenizers/wordpiece_tinysegmenter/vocab.txt > models/tokenizers/character/vocab.txt
```

Create pretraining data. It takes 3days.
```bash
$ ./create_pretraining_data.sh
```

Data files are created under `data/pretrain/wordpiece/` and `data/pretrain/character/`.

### Pretraining

Note: The published models were pretrained on Google's TPUs provided by [TPU Research Cloud](https://sites.research.google/trc/about/). The following procedure assumes Google Cloud environment.

Upload pretraining data files and config files to Google Cloud Storage.

Login to a GCE instance and prepare [original ALBERT](https://github.com/google-research/albert) repository. Original ALBERT models are trained on Tensorflow 1.5.  Tensorflow 1.5 requires Python 3.7 or below.
```bash
# Install Python 3.7 from source if you need
$ sudo apt update
$ sudo apt install -y build-essential tk-dev libncurses5-dev libncursesw5-dev libreadline6-dev libdb5.3-dev libgdbm-dev libsqlite3-dev libssl-dev libbz2-dev libexpat1-dev liblzma-dev zlib1g-dev libffi-dev libv4l-dev
$ wget https://www.python.org/ftp/python/3.7.12/Python-3.7.12.tgz
$ tar xzf Python-3.7.12.tgz
$ cd Python-3.7.12
$ ./configure --enable-optimizations --with-lto --enable-shared --prefix=/opt/python3.7 LDFLAGS=-Wl,-rpath,/opt/python3.7/lib
$ make -j 8
$ sudo make altinstall
$ cd

# Prepare original ALBERT repository
$ git clone https://github.com/google-research/albert
$ /opt/python3.7/bin/pip3.7 install -r albert/requirements.txt
$ /opt/python3.7/bin/pip3.7 install protobuf==3.20.0
```

Patch `albert/run_pretraining.py` to retrieve compressed pretraining data.
```
$ diff --git a/run_pretraining.py b/run_pretraining.py
index 949acc7..1c0e1d7 100644
--- a/run_pretraining.py
+++ b/run_pretraining.py
@@ -41,6 +41,10 @@ flags.DEFINE_string(
     "input_file", None,
     "Input TF example files (can be a glob or comma separated).")

+flags.DEFINE_string(
+    "compression_type", None,
+    "Compression type of input TF files (GZIP, ZLIB).")
+
 flags.DEFINE_string(
     "output_dir", None,
     "The output directory where the model checkpoints will be written.")
@@ -425,12 +429,12 @@ def input_fn_builder(input_files,
       # even more randomness to the training pipeline.
       d = d.apply(
           tf.data.experimental.parallel_interleave(
-              tf.data.TFRecordDataset,
+              lambda input_file: tf.data.TFRecordDataset(input_file, compression_type=FLAGS.compression_type),
               sloppy=is_training,
               cycle_length=cycle_length))
       d = d.shuffle(buffer_size=100)
     else:
-      d = tf.data.TFRecordDataset(input_files)
+      d = tf.data.TFRecordDataset(input_files, compression_type=FLAGS.compression_type)
       # Since we evaluate for a fixed number of steps we don't want to encounter
       # out-of-range exceptions.
       d = d.repeat()
```

Create TPU node with TPU sorfware version 1.5.5.

Run pretraining. It takes 17 days on TPU v2-8.
```bash
# train wordpiece model
$ export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
$ export WORK_DIR="gs://YOUR_BUCKET_NAME_FOR_WORDPIECE_MODEL"
$ export TPU_NAME=YOUR_TPU_NAME
$ /opt/python3.7/bin/python3.7 -u -m albert.run_pretraining \
    --input_file=${WORK_DIR}/data/pretrain_*.tfrecord.gz \
    --output_dir=${WORK_DIR}/model_base/ \
    --albert_config_file=${WORK_DIR}/configs/wordpiece_tinysegmenter_base.json \
    --do_train \
    --do_eval \
    --train_batch_size=512 \
    --eval_batch_size=64 \
    --max_seq_length=512 \
    --max_predictions_per_seq=20 \
    --optimizer='lamb' \
    --learning_rate=.00022 \
    --num_train_steps=1000000 \
    --num_warmup_steps=25000 \
    --save_checkpoints_steps=40000 \
    --use_tpu True \
    --tpu_name=${TPU_NAME} \
    --compression_type GZIP > train_wordpiece.log 2>&1 & disown

# train character model
$ export WORK_DIR="gs://YOUR_BUCKET_NAME_FOR_CHARACTER_MODEL"
$ export TPU_NAME=YOUR_TPU_NAME
$ /opt/python3.7/bin/python3.7 -u -m albert.run_pretraining \
    --input_file=${WORK_DIR}/data/pretrain_*.tfrecord.gz \
    --output_dir=${WORK_DIR}/model_base/ \
    --albert_config_file=${WORK_DIR}/configs/character_base.json \
    --do_train \
    --do_eval \
    --train_batch_size=512 \
    --eval_batch_size=64 \
    --max_seq_length=512 \
    --max_predictions_per_seq=20 \
    --optimizer='lamb' \
    --learning_rate=.00022 \
    --num_train_steps=1000000 \
    --num_warmup_steps=25000 \
    --save_checkpoints_steps=40000 \
    --use_tpu True \
    --tpu_name=${TPU_NAME} \
    --compression_type GZIP > train_character.log 2>&1 & disown
```

### Convert pretrained tensorflow checkpoints to transformers pytorch models

```bash
# convert wordpiece model
$ export WORK_DIR="gs://YOUR_BUCKET_NAME_FOR_WORDPIECE"
$ mkdir -p models/wordpiece_tinysegmenter/base
$ mkdir -p checkpoints/wordpiece_tinysegmenter/base
$ gsutil cp ${WORK_DIR}/configs/wordpiece_tinysegmenter_base.json models/wordpiece_tinysegmenter/base/config.json
$ gsutil cp ${WORK_DIR}/model_base/model.ckpt-best* checkpoints/wordpiece_tinysegmenter/base/
$ python convert_albert_original_tf_checkpoint_to_pytorch.py \
    --tf_checkpoint_path checkpoints/wordpiece_tinysegmenter/base/model.ckpt-best \
    --albert_config_file models/wordpiece_tinysegmenter/base/config.json \
    --pytorch_dump_path models/wordpiece_tinysegmenter/base/pytorch_model.bin

# convert character model
$ export WORK_DIR="gs://YOUR_BUCKET_NAME_FOR_CHARACTER"
$ mkdir -p models/character/base
$ mkdir -p checkpoints/character/base
$ gsutil cp ${WORK_DIR}/configs/character_base.json models/character/base/config.json
$ gsutil cp ${WORK_DIR}/model_base/model.ckpt-best* checkpoints/character/base/
$ python convert_albert_original_tf_checkpoint_to_pytorch.py \
    --tf_checkpoint_path checkpoints/character/base/model.ckpt-best \
    --albert_config_file models/character/base/config.json \
    --pytorch_dump_path models/character/base/pytorch_model.bin
```

## Related Works

### Original ALBERT models by Google Research Team

https://github.com/google-research/albert

### MeCab-tokenized Japanese BERT models by Tohoku University

https://github.com/cl-tohoku/bert-japanese

## Licenses

The pretrained models are distributed under the terms of the Creative Commons Attribution-ShareAlike 3.0.

The codes in this repository are distributed under the Apache License 2.0.

## Acknowledgments

The models are trained with Cloud TPUs provided by [TensorFlow Research Cloud](https://sites.research.google/trc/about/) program.
