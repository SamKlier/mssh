import logging
from pathlib import Path

from jpeg_mssh import JPEG_MSSH
from mssh_similarities import mssh_tversky_similarity, jaccard_similarity

if __name__ == '__main__':
    # Enter path to directory with exemplary JPEG files here
    db_path = Path(r"test_files")
    # Select n-gram size, 2 is recommended
    ngram_size = 2
    # Source SD for the Pixel 8 Pro:
    source_sd = 'c0c4c0dac4c4c4dac4dbd8e1d961d9ffdad9dbc0dbdbe0e2e1dbe1e0e1e1e1e2e2e1e2ebebc4|010001010101010f010f0110011001120112011a011a011b011b01280128013101310132013202130213876987698825'

    # Files in test_files directory have been captured with Pixel 4a or Pixel 8 Pro, as indicate by the name
    # Generate MSSH SDs per file
    jpeg_sds = []
    for mfile_path in db_path.rglob('*.jpg'):
        try:
            jpeg_src_hasher = JPEG_MSSH([mfile_path], ngram_size)
            mssh_sd = jpeg_src_hasher.get_str_hash()
            jpeg_sds.append((mfile_path, mssh_sd))
            print(f'{mfile_path}:\n\t{mssh_sd}')
        except Exception as e:
            logging.error(f'{e} for:\n\t{mfile_path}')

    # Compare results with the source SD:
    for res in jpeg_sds:
        print(f'{res[0]}:\n\t{mssh_tversky_similarity(source_sd, res[1])}')
