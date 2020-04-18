# Dmitry Kisler © 2020-present
# www.dkisler.com

import os
import pandas
import functools
from io import StringIO
from typing import Tuple, List, Union
import pickle


def parser_conllu_to_dict(document: str,
                          meta_description: dict = {}) -> List[dict]:
    """CoNLL-U file parser.
    
    See format details here: https://universaldependencies.org/format.html
    
    Args:
      document: Sentences string to parse.
      meta_description: Metadata tags.
                      If provided, it must contain 'tags' and 'delimiter', i.e.:
                      meta_description = {
                          "tags": ["tag1", "tag2"],
                          "delimiter": "=",
                      }
    
    Returns:
      Dict with sentences, UD tokens.
    """
    _comment_line = '#'
    _sentence_delimiter = '\n\n'
    _col_delimiter = '\t'
    _col_names = ["id", "form", "lemma",
                  "upos", "xpos",
                  "feats", "head", "deprel",
                  "deps", "misc"]
    _na_char = '_'
    _feat_delimiter = '='
    _feats_delimiter = '|'

    if meta_description:
        def _meta_extractor(string: str) -> str:
            """Helper function to extract meta data string."""
            return string.split(meta_description['delimiter'])[1].strip()

    def _feat_parser(feats: str) -> dict:
        """Feature parser from string into dict."""
        if feats == _na_char:
            return _na_char
        return {
            i[0]: i[1]
            for i in [feat.split(_feat_delimiter)[:2]
                      for feat in feats.split(_feats_delimiter)]
        }

    output = []

    for sentence in document.split(_sentence_delimiter):
        if sentence == "":
            break
        data = []
        meta = {}
        for line in sentence.split('\n'):
            if not line.startswith(_comment_line):
                data.append(line)
            else:
                if not meta_description:
                    continue
                else:
                    comment_tags = meta_description['tags'].copy()
                    for comment_tag in comment_tags:
                        if comment_tag in line:
                            meta[comment_tag] = _meta_extractor(line)
                            comment_tags.remove(comment_tag)

        data_str = '\n'.join(data)
        del data

        with StringIO(data_str.replace('"', '\\"')) as data:
            df = pandas.read_csv(data,
                                 delimiter=_col_delimiter,
                                 header=None,
                                 quotechar='"',
                                 names=_col_names)

        tokens = df.to_dict('records')
        for token in tokens:
            try:
                token['feats'] = _feat_parser(token['feats'])
            except Exception as ex:
                print(ex)
                continue

        out = {
            "tokens": tokens,
        }
        if meta:
            out['meta'] = meta

        output.append(out)

    return output


def corpus_parser(path: str) -> Union[Tuple[List[dict], None],
                                      Tuple[list, str]]:
    """Function to read corpus from conllu corpus file.
    
    Args:
      path: File path.
    
    Returns:
      List of token trees with error string in case of any.
    """
    if not os.path.isfile(path):
        return [], f"File {path} doesn't exist."
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            document = f.read()
        return parser_conllu_to_dict(document,
                                     meta_description={
                                         "tags": ["sent_id", "text", "s_tape"],
                                         "delimiter": "=",
                                         }
                                     ), None
    except Exception as ex:
        return [], ex


