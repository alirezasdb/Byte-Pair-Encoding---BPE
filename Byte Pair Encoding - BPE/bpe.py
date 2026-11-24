
from collections import Counter
import re


class BPETokenizer:
    """
    Character-level BPE tokenizer.

    Features:
    - Frequency-weighted BPE training
    - Whitespace preservation
    - Ordered merge operations
    - Vocabulary frequency statistics
    - Encode and decode
    """

    def __init__(self, num_merges=100, min_frequency=2):

        self.num_merges = num_merges
        self.min_frequency = min_frequency

        self.vocab = {}
        self.merges = []
        self.token_to_id = {}

        self.is_trained = False

    def _preprocess(self, text):
        return text.strip().split()

    def _get_pair_counts(self, word_freq):

        pair_counts = Counter()

        for symbols, frequency in word_freq.items():

            for i in range(len(symbols) - 1):

                pair = (symbols[i], symbols[i + 1])

                pair_counts[pair] += frequency

        return pair_counts

    def _merge_pair(self, word_freq, pair_to_merge):

        new_word_freq = Counter()

        first, second = pair_to_merge

        for symbols, frequency in word_freq.items():

            new_symbols = []
            i = 0

            while i < len(symbols):

                if (
                    i < len(symbols) - 1
                    and symbols[i] == first
                    and symbols[i + 1] == second
                ):

                    new_symbols.append(first + second)
                    i += 2

                else:

                    new_symbols.append(symbols[i])
                    i += 1

            new_word_freq[tuple(new_symbols)] += frequency

        return new_word_freq

    def train(self, corpus):

        word_counts = Counter()
        whitespace_counts = Counter()

        for sentence in corpus:

            word_counts.update(
                re.findall(r"\S+", sentence)
            )

            whitespace_counts.update(
                re.findall(r"\s+", sentence)
            )

        word_freq = Counter()

        for word, frequency in word_counts.items():

            symbols = tuple(word)

            word_freq[symbols] += frequency

        self.merges = []

        for step in range(self.num_merges):

            pair_counts = self._get_pair_counts(word_freq)

            if not pair_counts:
                break

            best_pair, best_frequency = pair_counts.most_common(1)[0]

            if best_frequency < self.min_frequency:
                break

            word_freq = self._merge_pair(
                word_freq,
                best_pair
            )

            self.merges.append(best_pair)

        # Build frequency-weighted vocabulary
        self.vocab = Counter()

        for symbols, frequency in word_freq.items():

            for symbol in symbols:

                self.vocab[symbol] += frequency

        # Preserve whitespace as independent tokens
        for whitespace, frequency in whitespace_counts.items():

            self.vocab[whitespace] += frequency

        self.vocab = dict(self.vocab)

        self.token_to_id = {
            token: idx
            for idx, token in enumerate(sorted(self.vocab))
        }

        self.is_trained = True

        return self

    def encode(self, text):

        if not self.is_trained:
            raise RuntimeError(
                "Tokenizer has not been trained."
            )

        encoded = []

        # Preserve exact whitespace
        pieces = re.findall(r"\s+|\S+", text)

        for piece in pieces:

            if piece.isspace():

                encoded.append(piece)
                continue

            symbols = list(piece)

            for first, second in self.merges:

                merged = []
                i = 0

                while i < len(symbols):

                    if (
                        i < len(symbols) - 1
                        and symbols[i] == first
                        and symbols[i + 1] == second
                    ):

                        merged.append(first + second)
                        i += 2

                    else:

                        merged.append(symbols[i])
                        i += 1

                symbols = merged

            encoded.extend(symbols)

        return encoded

    def decode(self, tokens):

        if not self.is_trained:
            raise RuntimeError(
                "Tokenizer has not been trained."
            )

        return "".join(tokens)

    def get_vocab(self):

        return self.vocab.copy()
