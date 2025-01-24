import logging
from pathlib import Path
import pandas as pd
from utils.jpeg_structure import JPEGTokenizer, JPEGAppHasher


class JPEG_MSSH:
    """
    A class for processing JPEG files and creating a Media Source Similarity Hash (MSSH).

    Attributes:
        source_files (list[Path, str]): List of paths to the source JPEG files.
        The hash will be generated based on all given files and not per file.
        ngram_size (int): Size of the n-grams.
    """

    def __init__(self, source_files: list[Path, str], ngram_size: int = 2):
        logging.debug(f'[JPEGSrcHasher] Hashing:\t{source_files}')
        self.shingle_size = ngram_size
        self.source_files = source_files

    def calculate_struct_ngrams(self, jpeg_tokens: pd.DataFrame) -> set:
        """
        Calculates n-grams based on JPEG structure tokens.

        Parameters:
            jpeg_tokens (DataFrame): The tokenized JPEG structure.

        Returns:
            set: A set of n-grams derived from structure tokens.
        """
        struct_tokens = jpeg_tokens['value']
        struct_tokens = struct_tokens.apply(lambda v: v[-2:]).tolist()
        logging.debug(f'[JPEG_MSSH] Received Structure Tokens: {struct_tokens}')
        struct_tokens = [tok[1:] for tok in struct_tokens]
        ngrams = self._make_ngrams(struct_tokens)
        return ngrams

    def calculate_app_ngrams(self, jpeg_tokens: pd.DataFrame, data: bytes, file_path: [Path, str]) -> set:
        """
        Calculates n-grams based on supported APP segments. Right now, only APP1 standard Exif segment is supported.

        Parameters:
            jpeg_tokens (DataFrame): The tokenized JPEG structure.
            data (bytes): The binary data of the JPEG file.
            file_path (Path or str): Path to the source JPEG file.

        Returns:
            set: A set of n-grams derived from the supported APP segments.
        """
        app_tokens = JPEGAppHasher.tokenize_apps(data, jpeg_tokens, file_path)
        logging.debug(f'[JPEG_MSSH] Received APP Tokens: {app_tokens}')
        ngrams = self._make_ngrams(app_tokens)
        return ngrams

    def _make_ngrams(self, tokens: list) -> set:
        """
        Converts a list of tokens into a set of n-grams.

        Parameters:
            tokens (list): List of tokens to be converted.

        Returns:
            set: The set of n-grams.
        """
        tokens = [v.hex() for v in tokens]
        n_gram_list = [''.join(tokens[i:i + self.shingle_size]) for i in range(len(tokens) - self.shingle_size + 1)]
        n_gram_set = set(n_gram_list)
        logging.debug(f'[JPEG_MSSH] Shingles:\t{n_gram_set}')
        return n_gram_set

    def get_hashes(self):
        """
         Generates a two part MSSH for the given list of JPEG files.

         Returns:
             list: A list containing two strings for each part of the hash.
         """
        app_ngrams = set()
        struct_ngrams = set()
        for source_file in self.source_files:
            with open(source_file, 'rb') as f:
                data = f.read()
                logging.debug(f'[JPEG_MSSH] read bytes from file: {len(data)}')
                jpeg_tokens = JPEGTokenizer(data).tokenize()
                logging.debug(f'[JPEG_MSSH] Received Tokens: {jpeg_tokens}')
                struct_ngrams.update(self.calculate_struct_ngrams(jpeg_tokens))
                app_ngrams.update(self.calculate_app_ngrams(jpeg_tokens, data, source_file))

        return [''.join(sorted(struct_ngrams)), ''.join(sorted(app_ngrams))]

    def get_str_hash(self, separator: str = '|'):
        """
         Generates a string representation of the two parted MSSH for the given list of JPEG files.

        Parameters:
            separator (str): Character that is used to separate the two hash parts.

         Returns:
             str: A single string that consists of the two hash parts, separated by the separator.
         """
        return separator.join(self.get_hashes())


