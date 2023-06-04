import collections
import copy
import os
import unicodedata
from typing import Optional

import tinysegmenter
from transformers.models.bert import BertTokenizer, WordpieceTokenizer
from transformers.models.bert_japanese.tokenization_bert_japanese import load_vocab


class BertJapaneseTinySegmenterTokenizer(BertTokenizer):
    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        do_word_tokenize=True,
        do_subword_tokenize=True,
        subword_tokenizer_type="wordpiece",
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        manual_subword_marking=False,
        **kwargs
    ):
        super(BertTokenizer, self).__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            do_lower_case=do_lower_case,
            do_word_tokenize=do_word_tokenize,
            do_subword_tokenize=do_subword_tokenize,
            word_tokenizer_type='tinysegmenter',
            subword_tokenizer_type=subword_tokenizer_type,
            never_split=never_split,
            **kwargs,
        )
        # ^^ We call the grandparent's init, not the parent's.

        self.manual_subword_marking = manual_subword_marking

        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

        self.do_word_tokenize = do_word_tokenize
        self.lower_case = do_lower_case
        self.never_split = never_split
        if do_word_tokenize:
            self.word_tokenizer = TinySegmenterTokenizer(do_lower_case=do_lower_case, never_split=never_split)

        self.do_subword_tokenize = do_subword_tokenize
        self.subword_tokenizer_type = subword_tokenizer_type
        if do_subword_tokenize:
            if subword_tokenizer_type == "wordpiece":
                self.subword_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)
            elif subword_tokenizer_type == "character":
                self.subword_tokenizer = CharacterTokenizer(
                    vocab=self.vocab,
                    unk_token=self.unk_token,
                    add_subword_markers=manual_subword_marking,
                    never_split=self.all_special_tokens
                )
            else:
                raise ValueError(f"Invalid subword_tokenizer_type '{subword_tokenizer_type}' is specified.")

    @property
    def do_lower_case(self):
        return self.lower_case

    def __getstate__(self):
        state = dict(self.__dict__)
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.word_tokenizer = TinySegmenterTokenizer(do_lower_case=self.do_lower_case, never_split=self.never_split)

    def _tokenize(self, text):
        if self.do_word_tokenize:
            tokens = self.word_tokenizer.tokenize(text, never_split=self.all_special_tokens)
        else:
            tokens = [text]

        if self.do_subword_tokenize:
            split_tokens = [sub_token for token in tokens for sub_token in self.subword_tokenizer.tokenize(token)]
        else:
            split_tokens = tokens

        return split_tokens


class TinySegmenterTokenizer:
    """Runs basic tokenization with tinysegmenter."""

    def __init__(
        self,
        do_lower_case=False,
        never_split=None,
        normalize_text=True,
    ):
        self.do_lower_case = do_lower_case
        self.never_split = never_split if never_split is not None else []
        self.normalize_text = normalize_text

    def tokenize(self, text, never_split=None):
        """Tokenizes a piece of text."""
        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)

        never_split = self.never_split + (never_split if never_split is not None else [])

        split_text = [text]
        for special_token in never_split:
            _parts = []
            for _text in split_text:
                _split = _text.split(special_token)
            _parts.append(_split[0])
            for i in range(1, len(_split)):
                _parts.append(special_token)
                _parts.append(_split[i])
            split_text = _parts

        tokens = []
        for _part in split_text:
            if _part in never_split:
                tokens.append(_part)
                continue
            for token in tinysegmenter.tokenize(_part):
                if self.do_lower_case:
                    token = token.lower()
                tokens.append(token)

        return tokens


class CharacterTokenizer:
    """Runs Character tokenization."""

    def __init__(self, vocab, unk_token, normalize_text=True, add_subword_markers=False, never_split=[]):
        """
        Constructs a CharacterTokenizer.

        Args:
            **vocab**:
                Vocabulary object.
            **unk_token**: str
                A special symbol for out-of-vocabulary token.
            **normalize_text**: (`optional`) boolean (default True)
                Whether to apply unicode normalization to text before tokenization.
            **add_subword_marker**: (optional`) boolean (default False)
                If set to True, the subword marker "##" will be prepended to i-th (i > 0) characters within each word.
        """
        self.vocab = vocab
        self.unk_token = unk_token
        self.normalize_text = normalize_text
        self.add_subword_markers = add_subword_markers
        self.never_split = never_split

    def tokenize(self, text):
        """Tokenizes a piece of text into characters.
        For example:
            input = "apple"
            output = ["a", "##p", "##p", "##l", "##e"]  (if add_subword_markers is True)
                     ["a", "p", "p", "l", "e"]          (if add_subword_markers is False)
        Args:
            text: A single token or whitespace separated tokens.
                This should have already been passed through `BasicTokenizer`.
        Returns:
            A list of characters.
        """
        if text in self.never_split:
            return [text]

        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)

        output_tokens = []
        for i, char in enumerate(text):
            if char not in self.vocab:
                output_tokens.append(self.unk_token)
                continue

            if self.add_subword_markers and i > 0:
                output_tokens.append("##{}".format(char))
            else:
                output_tokens.append(char)

        return output_tokens
