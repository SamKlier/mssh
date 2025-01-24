import numpy as np


def split_hash(hash: str, ngram_size: int):
    """
    Splits a hash into n-grams.

    Parameters:
        hash (str): The input hash to be split.
        ngram_size (int): The size of the n-grams.

    Returns:
        set: A set of n-grams extracted from the hash.
    """
    ngrams = set()
    for i in range(0, len(hash), ngram_size):
        ngrams.add(hash[i:i + ngram_size])
    return ngrams


def jaccard_similarity(sd1: str, sd2: str, ngram_size: int = 2, separator: str = '|'):
    """
    Calculates the Jaccard similarity between two similarity digests.

    Parameters:
        sd1 (str): The first hash. Order of hashes does not matter.
        sd2 (str): The second hash. Order of hashes does not matter.
        ngram_size (int): The size of the n-grams used for SD generation (default: 2).
        separator (str): Separator used to split the JPEG/APP1 parts (default: '|').

    Returns:
        float: Jaccard similarity value [0, 1]. The higher, the more similar.
        A value of 1 does not indicate that sources are identical.
    """
    sd1 = sd1.upper()
    sd2 = sd2.upper()
    spl_h1 = sd1.split(separator)
    spl_h2 = sd2.split(separator)

    # 2 Bytes per JPEG tag
    ngrams_1 = split_hash(spl_h1[0], ngram_size * 2)
    ngrams_2 = split_hash(spl_h2[0], ngram_size * 2)

    if len(spl_h1) > 1 and len(spl_h2) > 1:
        # 4 Bytes per APP1 tag
        ngrams_1.update(split_hash(spl_h1[1], ngram_size * 4))
        ngrams_2.update(split_hash(spl_h2[1], ngram_size * 4))

    jaccard = float(len(ngrams_1.intersection(ngrams_2)) / len(ngrams_1.union(ngrams_2)))

    return jaccard


def mssh_tversky_similarity(source_sd: str, media_sd: str, ngram_size: int = 2, separator: str = '|'):
    """
    Calculates the Tversky similarity index between the similarity digest of a source and a media file.

    Parameters:
        source_sd (str): The source SD. ATTENTION: Order does matter!
        media_sd (str): The meddia file SD. ATTENTION: Order does matter!
        ngram_size (int): The size of the n-grams used for SD generation (default: 2).
        separator (str): Separator used to split the JPEG/APP1 parts (default: '|').

    Returns:
        float: Tversky similarity index [0, 1]. The higher, the more similar.
        A value of 1 does not indicate that sources are identical.
    """
    source_sd = source_sd.upper()
    media_sd = media_sd.upper()

    source_sd = source_sd.split(separator)
    media_sd = media_sd.split(separator)

    # 2 Bytes per JPEG tag
    ngrams_src = split_hash(source_sd[0], ngram_size * 2)
    ngrams_mf = split_hash(media_sd[0], ngram_size * 2)

    if len(source_sd) > 1 and len(media_sd) > 1:
        # 4 Bytes per APP1 tag
        ngrams_src.update(split_hash(source_sd[1], ngram_size * 4))
        ngrams_mf.update(split_hash(media_sd[1], ngram_size * 4))

    intersection_size = len(ngrams_src & ngrams_mf)
    overlap_size = len(ngrams_mf - ngrams_src)

    try:
        beta = 1
        # alpha = 0
        tversky_index = intersection_size / (intersection_size + beta * overlap_size)
        return tversky_index
    except ZeroDivisionError:
        return np.nan
