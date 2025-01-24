import struct


def check_sof_length(data, address):
    """
    Checks if the Start of Frame (SOF) length matches the expected value.

    Parameters:
        data (bytes): Binary JPEG data.
        address (int): Address of the SOF marker in the data.

    Returns:
        bool: True if the length matches the expected value, False otherwise.
    """
    length = (data[address + 2] << 8) + data[address + 3]
    comp_in_frame = data[address + 9]
    exp_length = 8 + 3 * comp_in_frame
    return length == exp_length


def check_sos_length(data, address):
    """
    Checks if the Start of Scan (SOS) length matches the expected value.

    Parameters:
        data (bytes): Binary JPEG data.
        address (int): Address of the SOS marker in the data.

    Returns:
        bool: True if the length matches the expected value, False otherwise.
    """
    length = (data[address + 2] << 8) + data[address + 3]
    comp_in_scan = data[address + 4]
    exp_length = 6 + comp_in_scan * 2
    return length == exp_length


def get_app_length(data, address):
    """
    Retrieves the length of an APP segment.

    Parameters:
        data (bytes): Binary JPEG data.
        address (int): Address of the APP marker in the data.

    Returns:
        int: The length of the APP segment.
    """
    length = (data[address + 2] << 8) + data[address + 3]
    return length


def decode_sof(data, index):
    """
    Decodes and prints details from a Start of Frame (SOF) marker.

    Parameters:
        data (bytes): Binary JPEG data.
        index (int): Address of the SOF marker in the data.
    """
    # SOF marker is followed by a 2-byte length field
    length = (data[index + 2] << 8) + data[index + 3]
    print(f"\nLength: {length}")
    start = index + 4

    # Frame header fields
    precision = data[start]
    height = (data[start + 1] << 8) + data[start + 2]
    width = (data[start + 3] << 8) + data[start + 4]
    num_components = data[start + 5]

    print(f"Precision: {precision} bits")
    print(f"Image Width: {width}")
    print(f"Image Height: {height}")
    print(f"Number of Components: {num_components}")

    components = []
    for i in range(num_components):
        comp_id = data[start + 6 + 3 * i]
        samp_factor = data[start + 7 + 3 * i]
        quant_table = data[start + 8 + 3 * i]
        components.append({
            'Component ID': comp_id,
            'Sampling Factors': samp_factor,
            'Quantization Table': quant_table
        })

    for comp in components:
        print(f"Component ID: {comp['Component ID']}, "
              f"Sampling Factors: {comp['Sampling Factors']}, "
              f"Quantization Table: {comp['Quantization Table']}")


def decode_sos(data, index):
    """
    Decodes and prints details from a Start of Scan (SOS) marker.

    Parameters:
        data (bytes): Binary JPEG data.
        index (int): Address of the SOS marker in the data.

    Returns:
        bytes: The actual compressed data following the SOS marker.
    """
    # SOS marker is followed by a 2-byte length field
    length = (data[index + 2] << 8) + data[index + 3]
    print(f"\nLength: {length}")
    start = index + 4

    # Number of components in scan
    num_components = data[start]
    print(f"Number of Components in Scan: {num_components}")

    components = []
    for i in range(num_components):
        comp_id = data[start + 1 + 2 * i]
        huffman_table = data[start + 2 + 2 * i]
        components.append({
            'Component ID': comp_id,
            'Huffman Table': huffman_table
        })

    for comp in components:
        print(f"Component ID: {comp['Component ID']}, "
              f"Huffman Table: {comp['Huffman Table']}")

    # Scan header specific fields
    start_spectral_selection = data[start + 1 + 2 * num_components]
    end_spectral_selection = data[start + 2 + 2 * num_components]
    successive_approximation = data[start + 3 + 2 * num_components]

    print(f"Start of Spectral Selection: {start_spectral_selection}")
    print(f"End of Spectral Selection: {end_spectral_selection}")
    print(f"Successive Approximation: {successive_approximation}")

    # The actual compressed data starts immediately after the SOS marker header
    compressed_data_start = start + 4 + 2 * num_components
    compressed_data = data[compressed_data_start:]

    return compressed_data


def read_jpeg_huffman_table(jpeg_file):
    """
    Reads and prints details of the Huffman Table (DHT) from a JPEG file.

    Parameters:
        jpeg_file (str): Path to the JPEG file.
    """
    with open(jpeg_file, 'rb') as f:
        # Read the JPEG file header
        data = f.read()

    # Search for the DHT marker (0xFFC4)
    dht_index = data.find(b'\xFF\xC4')

    if dht_index == -1:
        print("No Huffman Table (DHT) found in the file.")
        return

    # Skip marker (2 bytes) and length of segment (2 bytes)
    segment_length = struct.unpack('>H', data[dht_index + 2:dht_index + 4])[0]
    dht_data = data[dht_index + 4:dht_index + 2 + segment_length]

    # The first byte in DHT segment provides table class and identifier (e.g., DC/AC and table ID)
    table_class_and_id = dht_data[0]
    table_class = (table_class_and_id >> 4) & 0x0F  # DC = 0, AC = 1
    table_id = table_class_and_id & 0x0F

    print(f"Huffman Table Class: {table_class}, Table ID: {table_id}")

    # Next 16 bytes indicate the number of codes for each bit-length (1 to 16 bits)
    lengths = dht_data[1:17]
    symbols = dht_data[17:]

    print("Lengths of Huffman codes (1-16 bits):", lengths)
    #print("Huffman symbols:", symbols)


def read_jpeg_quantization_table(jpeg_file):
    """
    Reads and prints details of the Quantization Table (DQT) from a JPEG file.

    Parameters:
        jpeg_file (str): Path to the JPEG file.
    """
    with open(jpeg_file, 'rb') as f:
        # Read the JPEG file content
        data = f.read()

    # Search for the DQT marker (0xFFDB)
    dqt_index = data.find(b'\xFF\xDB')

    if dqt_index == -1:
        print("No Quantization Table (DQT) found in the file.")
        return

    # Skip marker (2 bytes) and length of segment (2 bytes)
    segment_length = struct.unpack('>H', data[dqt_index + 2:dqt_index + 4])[0]
    dqt_data = data[dqt_index + 4:dqt_index + 2 + segment_length]

    # First byte in DQT segment gives the precision (0 = 8-bit, 1 = 16-bit) and table ID
    precision_and_id = dqt_data[0]
    precision = (precision_and_id >> 4) & 0x0F  # Precision: 0 (8-bit) or 1 (16-bit)
    table_id = precision_and_id & 0x0F  # Table ID: 0-3

    print(f"Quantization Table ID: {table_id}, Precision: {'8-bit' if precision == 0 else '16-bit'}")

    # Based on precision, the next 64 or 128 bytes are the quantization table values
    if precision == 0:
        # 8-bit quantization values (64 bytes)
        quant_table = list(dqt_data[1:65])
    else:
        # 16-bit quantization values (128 bytes)
        quant_table = [struct.unpack('>H', dqt_data[i:i + 2])[0] for i in range(1, 129, 2)]

    print("Quantization Table Values:")
    print(quant_table)