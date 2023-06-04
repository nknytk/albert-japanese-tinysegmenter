import os
from typing import List, Optional, Tuple

from tokenizers import NormalizedString, PreTokenizedString
import tinysegmenter

Offsets = Tuple[int, int]


class TinySegmenterPreTokenizer:
    def __init__(self):
        pass

    def split(self, i: int, text: NormalizedString) -> List[NormalizedString]:
        tokens = []
        cursor = 0
        text_str = str(text)
        for token in tinysegmenter.tokenize(text_str):
            start = text_str.index(token, cursor)
            end = start + len(token)
            tokens.append(text[start:end])
            cursor = end

        return tokens

    def pre_tokenize(self, text: PreTokenizedString):
        text.split(self.split)
