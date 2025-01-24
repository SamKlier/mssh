from typing import Literal

import pandas as pd


def get_tiff(data: bytes):
    """
    Extracts the TIFF header and determines byte order.

    Parameters:
        data (bytes): Binary data from the image file.

    Returns:
        tuple: A tuple containing the TIFF header, byte order, and offset to IFD0.

    Raises:
        ValueError: If no valid TIFF header is found.
    """
    magic = b'\x49\x49\x2a\x00\x08\x00\x00\x00'
    header_index = data.find(magic)
    if header_index != -1:
        byteorder = 'little'
    else:
        magic = b'\x4d\x4d\x00\x2a\x00\x00\x00\x08'
        header_index = data.find(magic)
        if header_index == -1:
            raise ValueError('No TIFF Header magic found.')
        byteorder = 'big'

    tiff_header = data[header_index:]
    off_ifd0 = len(magic)
    if off_ifd0 == 0:
        raise ValueError(f'Offset of IFD0 is 0! Magic: {magic}')

    return tiff_header, byteorder, off_ifd0


def extract_IFD(ifd: bytes, byteorder: Literal["little", "big"]):
    """
    Extracts Image File Directory (IFD) entries from TIFF data.

    Parameters:
        ifd (bytes): The IFD data segment.
        byteorder (str): Byte order ('little' or 'big').

    Returns:
        tuple: A tuple containing a DataFrame of extracted fields, the number of fields, and the offset to the next IFD.

    Raises:
        ValueError: If the number of fields is too large.
    """
    number_of_fields = int.from_bytes(ifd[:2], byteorder=byteorder)
    if number_of_fields > 256:
        raise ValueError(f'Number of Fields to big: {number_of_fields} for IFD ({byteorder})\n\t{ifd[:20].hex()}')

    fields = ifd[2:]
    fields = [fields[i:i + 12] for i in range(0, number_of_fields * 12, 12)]
    data = []
    for f in fields:
        if byteorder == 'little':
            id_bytes = f[:2].hex(' ').split(' ')
            id_bytes.reverse()
            tag_id = bytes.fromhex(''.join(id_bytes))
        else:
            tag_id = f[:2]

        d = {'value': tag_id,
             'type': int.from_bytes(f[2:4], byteorder=byteorder),
             'length': int.from_bytes(f[4:8], byteorder=byteorder),
             'payload': f[8:12]  # Raw! Endianness not factored in!!!
             }
        data.append(d)

    next_ifd_off = 2 + number_of_fields * 12
    next_ifd_address = int.from_bytes(ifd[next_ifd_off:next_ifd_off + 4], byteorder=byteorder)
    '''
    if next_ifd_address != 0:
        next_ifd_address = next_ifd_address + 12
    '''
    data = pd.DataFrame(data)
    return data, number_of_fields, next_ifd_address


def get_value_of_field(field: dict, byteorder: str):
    """
    Retrieves the value of a field based on its type and byte order.

    Parameters:
        field (dict): The field dictionary containing type and value.
        byteorder (str): Byte order ('little' or 'big').

    Returns:
        Union[int, bytes]: The extracted value based on the field type.

    Raises:
        NotImplementedError: If the field type is not implemented.
    """
    type = field['type']
    match type:
        case 4:
            val = int.from_bytes(field['value'], byteorder=byteorder, signed=False)
        case 7:
            val = field['value']
        case _:
            raise NotImplementedError(f'Case for type ({type}) is not implemented.')
    return val


def get_maker_off(data: bytes):
    """
    NOTE: Maker Notes not used.
    Determines the offset and type of Maker Note metadata in the image file.

    Parameters:
        data (bytes): Binary data from the image file.

    Returns:
        tuple: A tuple containing the offset and the detected maker note type.

    Raises:
        NotImplementedError: If the maker note type is unknown.
    """
    if data.find(b'Apple iOS') != -1:
        return 14, 'Apple iOS'
    if data.find(b'MOT\0') != -1:
        return 8, 'MOT'
    if data.find(b'\x07\x00\x01\x00\x07\x00\x04\x00\x00\x000100\x02\x00\x04') != -1:
        return -1, 'Samsung_Unknown'
    else:
        raise NotImplementedError(f'Unknown Maker Note:\n\t{data[:50]}\n\t{data[:50].hex()}\n')

