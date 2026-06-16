"""
tokenizer.py -- map Semantic IDs to/from a flat token vocabulary.

The RQ-VAE gives every item a tuple of digits, e.g. (37, 198, 4, 0) with
D = rq_levels + 1 positions (the last is the collision-disambiguation digit).
To feed these to a transformer we flatten user histories into one long token
stream, BUT we give each *position* its own slice of the vocabulary:

    [PAD] [BOS] | position-0 codes | position-1 codes | ... | position-(D-1) codes

So the same integer 37 means different things at position 0 vs position 1 -- the
model never confuses a coarse code with a fine one. Two special tokens (PAD,
BOS) sit at the front.

A user history [i_1, i_2, ...] becomes:

    BOS  d0(i_1) d1(i_1) ... d0(i_2) d1(i_2) ...

and predicting the next item means generating its D tokens in order.
"""

from config import cfg


class SemanticTokenizer:
    PAD = 0
    BOS = 1
    N_SPECIAL = 2

    def __init__(self, item_codes):
        # item_codes: list (len n_items) of D-length tuples of ints
        self.item_codes = [tuple(c) for c in item_codes]
        self.D = len(self.item_codes[0])
        # cardinality of each position = (max code seen) + 1
        self.card = [max(c[p] for c in self.item_codes) + 1 for p in range(self.D)]
        self.offset = [self.N_SPECIAL]
        for p in range(1, self.D):
            self.offset.append(self.offset[-1] + self.card[p - 1])
        self.vocab_size = self.offset[-1] + self.card[-1]
        # reverse map: code tuple -> item id (codes are unique after disambiguation)
        self.code2item = {c: i for i, c in enumerate(self.item_codes)}

    # ---- item <-> tokens -------------------------------------------------
    def item_tokens(self, item_id):
        c = self.item_codes[item_id]
        return [self.offset[p] + c[p] for p in range(self.D)]

    def tokens_to_item(self, tokens):
        """Inverse of item_tokens; returns item id or None if invalid tuple."""
        code = tuple(tokens[p] - self.offset[p] for p in range(self.D))
        return self.code2item.get(code)

    # ---- sequences -------------------------------------------------------
    def encode_sequence(self, items, add_bos=True):
        toks = [self.BOS] if add_bos else []
        for it in items:
            toks.extend(self.item_tokens(it))
        return toks

    # ---- constrained decoding helpers -----------------------------------
    def valid_token_range(self, position):
        """[lo, hi) of legal token ids for a given digit position (0..D-1)."""
        lo = self.offset[position]
        return lo, lo + self.card[position]


def build_tokenizer(item_codes):
    return SemanticTokenizer(item_codes)
