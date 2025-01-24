import logging
from enum import Enum, auto

import pandas as pd
from aenum import MultiValueEnum
from fastcrc import crc16
from lxml import etree

from utils import jpeg_utils, parse_tiff


class JPEGMarkers(Enum):
    """
    Enumeration of JPEG markers with their byte representations.
    """
    DHT = b'\xff\xc4'  # Define Huffman Table
    JPG = b'\xff\xc8'  # Reserved for JPEG extensions
    DAC = b'\xff\xcc'  # Define Arithmetic Coding
    RST0 = b'\xff\xd0'  # Restart Interval Termination 0
    RST1 = b'\xff\xd1'  # Restart Interval Termination 1
    RST2 = b'\xff\xd2'  # Restart Interval Termination 2
    RST3 = b'\xff\xd3'  # Restart Interval Termination 3
    RST4 = b'\xff\xd4'  # Restart Interval Termination 4
    RST5 = b'\xff\xd5'  # Restart Interval Termination 5
    RST6 = b'\xff\xd6'  # Restart Interval Termination 6
    RST7 = b'\xff\xd7'  # Restart Interval Termination 7
    SOI = b'\xff\xd8'  # Start of Image
    EOI = b'\xff\xd9'  # End of Image
    SOS = b'\xff\xda'  # Start of Scan
    DQT = b'\xff\xdb'  # Define Quantization Table
    DNL = b'\xff\xdc'  # Define Number of Lines
    DRI = b'\xff\xdd'  # Define Restart Interval
    DHP = b'\xff\xde'  # Define Hierarchical Progression
    EXP = b'\xff\xdf'  # Expand Reference Components
    APP0 = b'\xff\xe0'  # Application Segment 0 (usually JFIF)
    APP1 = b'\xff\xe1'  # Application Segment 1 (usually Exif)
    APP2 = b'\xff\xe2'  # Application Segment 2
    APP3 = b'\xff\xe3'  # Application Segment 3
    APP4 = b'\xff\xe4'  # Application Segment 4
    APP5 = b'\xff\xe5'  # Application Segment 5
    APP6 = b'\xff\xe6'  # Application Segment 6
    APP7 = b'\xff\xe7'  # Application Segment 7
    APP8 = b'\xff\xe8'  # Application Segment 8
    APP9 = b'\xff\xe9'  # Application Segment 9
    APP10 = b'\xff\xea'  # Application Segment 10
    APP11 = b'\xff\xeb'  # Application Segment 11
    APP12 = b'\xff\xec'  # Application Segment 12
    APP13 = b'\xff\xed'  # Application Segment 13
    APP14 = b'\xff\xee'  # Application Segment 14 (usually Adobe)
    APP15 = b'\xff\xef'  # Application Segment 15
    JPG0 = b'\xff\xf0'  # JPEG Extensions
    JPG1 = b'\xff\xf1'  # JPEG Extensions
    JPG2 = b'\xff\xf2'  # JPEG Extensions
    JPG3 = b'\xff\xf3'  # JPEG Extensions
    JPG4 = b'\xff\xf4'  # JPEG Extensions
    JPG5 = b'\xff\xf5'  # JPEG Extensions
    JPG6 = b'\xff\xf6'  # JPEG Extensions
    JPG7 = b'\xff\xf7'  # JPEG Extensions
    JPG8 = b'\xff\xf8'  # JPEG Extensions
    JPG9 = b'\xff\xf9'  # JPEG Extensions
    JPG10 = b'\xff\xfa'  # JPEG Extensions
    JPG11 = b'\xff\xfb'  # JPEG Extensions
    JPG12 = b'\xff\xfc'  # JPEG Extensions
    JPG13 = b'\xff\xfd'  # JPEG Extensions
    COM = b'\xff\xfe'  # Comment
    TEM = b'\xff\x01'  # Temporary marker
    SOF0 = b'\xff\xc0'  # Start of Frame (Baseline DCT)
    SOF1 = b'\xff\xc1'  # Start of Frame (Extended Sequential DCT)
    SOF2 = b'\xff\xc2'  # Start of Frame (Progressive DCT)
    SOF3 = b'\xff\xc3'  # Start of Frame (Lossless Sequential)
    SOF5 = b'\xff\xc5'  # Start of Frame (Differential Sequential DCT)
    SOF6 = b'\xff\xc6'  # Start of Frame (Differential Progressive DCT)
    SOF7 = b'\xff\xc7'  # Start of Frame (Differential Lossless Sequential)
    SOF9 = b'\xff\xc9'  # Start of Frame (Extended Sequential DCT, arithmetic coding)
    SOF10 = b'\xff\xca'  # Start of Frame (Progressive DCT, arithmetic coding)
    SOF11 = b'\xff\xcb'  # Start of Frame (Lossless Sequential, arithmetic coding)
    SOF13 = b'\xff\xcd'  # Start of Frame (Differential Sequential DCT, arithmetic coding)
    SOF14 = b'\xff\xce'  # Start of Frame (Differential Progressive DCT, arithmetic coding)
    SOF15 = b'\xff\xcf'  # Start of Frame (Differential Lossless Sequential, arithmetic coding)
    SLACK = None


class MarkerTypes(MultiValueEnum):
    """
    MultiValueEnum for categorizing JPEG marker types.
    """
    SOI = JPEGMarkers.SOI
    EOI = JPEGMarkers.EOI
    DRI = JPEGMarkers.DRI
    APP = JPEGMarkers.APP0, JPEGMarkers.APP1, JPEGMarkers.APP2, JPEGMarkers.APP3, JPEGMarkers.APP4, JPEGMarkers.APP5, \
          JPEGMarkers.APP6, JPEGMarkers.APP7, JPEGMarkers.APP8, JPEGMarkers.APP9, JPEGMarkers.APP10, JPEGMarkers.APP11, \
          JPEGMarkers.APP12, JPEGMarkers.APP13, JPEGMarkers.APP14, JPEGMarkers.APP15
    DQT = JPEGMarkers.DQT
    DHT = JPEGMarkers.DHT
    SOF = JPEGMarkers.SOF0, JPEGMarkers.SOF1, JPEGMarkers.SOF2, JPEGMarkers.SOF3, JPEGMarkers.SOF5, JPEGMarkers.SOF6, \
          JPEGMarkers.SOF7, JPEGMarkers.SOF9, JPEGMarkers.SOF10, JPEGMarkers.SOF11, JPEGMarkers.SOF13, \
          JPEGMarkers.SOF14, JPEGMarkers.SOF15
    SOS = JPEGMarkers.SOS
    RST = JPEGMarkers.RST0, JPEGMarkers.RST1, JPEGMarkers.RST2, JPEGMarkers.RST3, JPEGMarkers.RST4, JPEGMarkers.RST5, \
          JPEGMarkers.RST6, JPEGMarkers.RST7
    OTHER = JPEGMarkers.SLACK

    @staticmethod
    def get_type(marker):
        try:
            return MarkerTypes(marker)
        except ValueError:
            return MarkerTypes.OTHER


class JPEGStates(Enum):
    """
    Enumeration of JPEG parsing states.
    """
    INIT = auto()
    IMAGE = auto()
    FRAME = auto()
    SCAN = auto()

    @staticmethod
    def accepted_markers(state):
        """
        Returns the list of accepted markers for a given JPEG state.

        Parameters:
            state (JPEGStates): The current parsing state.

        Returns:
            list: List of accepted marker types.
        """
        match state:
            case JPEGStates.INIT:
                return [MarkerTypes.SOI]
            case JPEGStates.IMAGE:
                return [MarkerTypes.SOI, MarkerTypes.APP, MarkerTypes.DHT, MarkerTypes.DQT, MarkerTypes.DRI,
                        MarkerTypes.SOF, MarkerTypes.OTHER]
            case JPEGStates.FRAME:
                return [MarkerTypes.APP, MarkerTypes.DHT, MarkerTypes.DQT, MarkerTypes.DRI, MarkerTypes.SOF,
                        MarkerTypes.SOS]
            case JPEGStates.SCAN:
                return [MarkerTypes.DHT, MarkerTypes.RST, MarkerTypes.EOI]
            case _:
                raise NotImplementedError(f'Accepted Markers for State:\t{state}')


class JPEGTokenizer:
    """
    A tokenizer for extracting (valid) JPEG structure tokens.
    """
    skip_until = 0

    def __init__(self, data: bytes, debug: bool = False):
        """
        Initializes the JPEGTokenizer.

        Parameters:
            data (bytes): The binary data of the JPEG file.
            debug (bool): Whether to enable debugging output.
        """
        self.debug = debug
        self.data = data
        if self.data[:2] != JPEGMarkers.SOI.value:
            raise ValueError(f'No proper JPEG, does not start with SOI (instead: {self.data[:20]}.')
        self.tokens: pd.DataFrame = pd.DataFrame()
        self.end_of_jpeg_reached = False

    def tokenize(self):
        """
        Tokenizes the JPEG data into a DataFrame of markers and metadata.

        Returns:
            DataFrame: The tokenized JPEG structure. Note: only valid Tokens are returned.
        """
        self.tokens: pd.DataFrame = pd.DataFrame()
        self.end_of_jpeg_reached = False
        results = {'token': [], 'type': [], 'address': [], 'value': []}

        for marker in JPEGMarkers:
            start = 0
            while True:
                if not marker.value:
                    break
                pos = self.data.find(marker.value, start)
                if pos == -1:
                    break
                m_type = MarkerTypes.get_type(marker)
                results['token'].append(marker)
                results['value'].append(marker.value)
                results['type'].append(m_type)
                results['address'].append(pos)
                start = pos + 1

        df = pd.DataFrame(results)
        df.sort_values(by=['address'], inplace=True)
        if self.debug:
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            print(f'Raw results:\n{df}')

        validated = self._validate_results(df)
        eoi_address = validated['address'].iloc[-1]
        slack = self._check_slack(eoi_address)
        if slack is not None:
            self.tokens = pd.concat([validated, pd.DataFrame({'token': [JPEGMarkers.SLACK], 'type': [MarkerTypes.OTHER],
                                                              'address': [eoi_address + 2], 'value': [slack]})])
        else:
            self.tokens = validated

        if self.debug:
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            print(f'Validated results:\n{self.tokens}')
        return self.tokens

    def _validate_results(self, to_validate: pd.DataFrame) -> pd.DataFrame:
        """
        Validates the tokenized results against JPEG parsing states.
        Hence, e.g. a JPEGMarker besides the RST marker in a scan segment is not valid.

        Parameters:
            to_validate (DataFrame): The tokenized data to validate.

        Returns:
            DataFrame: The validated tokenized data.
        """
        state_stack = [JPEGStates.INIT, ]
        val_results = to_validate.apply(lambda entry: self._transition(state_stack, entry['type'], entry['address']),
                                        axis=1, result_type='expand')
        validated = to_validate.loc[val_results]
        if len(state_stack) != 1 and self.debug:
            print(f'State Stack not empty:\n\t{state_stack}')
        return validated

    def _check_slack(self, eoi_address):
        """
        Checks for slack bytes after the end of the JPEG.

        Parameters:
            eoi_address (int): Address of the EOI marker.

        Returns:
            bytes or None: Slack bytes if found, otherwise None.
        """
        slack_marker = self.data[eoi_address + 2:eoi_address + 3]
        if slack_marker == b'':
            return None
        return b'ff' + slack_marker

    def _transition(self, state_stack: list, m_type: MarkerTypes, address=None):
        """
        Handles state transitions for parsing the JPEG structure.

        Parameters:
            state_stack (list): The stack of current parsing states.
            m_type (MarkerTypes): The marker type to transition with.
            address (int, optional): Address of the marker.

        Returns:
            bool: Whether the transition was successful.
        """
        if address < self.skip_until:
            return False
        if self.end_of_jpeg_reached:
            return False

        # Check if marker is accepted in the actual state...
        if m_type not in JPEGStates.accepted_markers(state_stack[-1]):
            if self.debug:
                print(f'Rejecting {m_type} for {state_stack[-1]} @ {hex(int(address))}\n({state_stack})')
            return False

        match m_type:
            case MarkerTypes.SOI:
                state_stack.append(JPEGStates.IMAGE)
            case MarkerTypes.SOF:
                is_valid = jpeg_utils.check_sof_length(self.data, address)
                if is_valid:
                    state_stack.append(JPEGStates.FRAME)
                else:
                    return False
            case MarkerTypes.SOS:
                if jpeg_utils.check_sos_length(self.data, address):
                    state_stack.append(JPEGStates.SCAN)
                else:
                    return False
            case MarkerTypes.EOI:
                popped = state_stack.pop()
                while popped != JPEGStates.IMAGE:
                    popped = state_stack.pop()

                if len(state_stack) == 1:
                    self.end_of_jpeg_reached = True
                    # Warn if end of JPEG is found before 1MiB is reached.
                    if self.debug and address // 1024 // 1024 < 1:
                        print(f'End of JPEG reached @ {hex(int(address))}')
            case MarkerTypes.APP:
                self.skip_until = address + jpeg_utils.get_app_length(self.data, address)
            case MarkerTypes.DQT | MarkerTypes.DHT | MarkerTypes.DRI | MarkerTypes.RST | MarkerTypes.OTHER:
                pass
            case _:
                raise NotImplementedError(f'State:\t{state_stack[-1]}\tMarker:\t{m_type}')

        return True

    def print_struct(self):
        for t in self.tokens['token']:
            print(t.name)


class JPEGAppHasher:
    """
    A utility class for hashing JPEG APP segments.
    """

    def __init__(self, file_path: str, ngram_size: int = 2, debug: bool = False):
        """
        Initializes the JPEGAppHasher.

        Parameters:
            file_path (str): Path to the JPEG file.
            ngram_size (int): Size of the shingles (default: 2).
            debug (bool): Whether to enable debugging output.
        """
        self.debug = debug
        if self.debug:
            print(f'\nHashing:\t{file_path}')
        self.shingle_size = ngram_size
        with open(file_path, 'rb') as f:
            self.data = f.read()
        self.jpeg_tokens = JPEGTokenizer(self.data, debug=debug).tokenize()

    @staticmethod
    def tokenize_apps(data: bytes, jpeg_struct_tokens: pd.DataFrame, file_path: str = '') -> list:
        """
        Tokenizes APP segments from a JPEG file.
        Todo: only APP1 Exif data handled right now.

        Parameters:
            data (bytes): Binary JPEG data.
            jpeg_struct_tokens (DataFrame): Tokenized JPEG structure.
            file_path (str, optional): Path to the JPEG file.

        Returns:
            list: List of tokens derived from APP segments.
        """
        app1_segments = jpeg_struct_tokens.loc[jpeg_struct_tokens['token'] == JPEGMarkers.APP1]
        app1_tokens = []
        for app1_address in app1_segments['address']:
            app1_tokens.extend(JPEGAppHasher._tokenize_app1(data, app1_address, file_path))
        return app1_tokens

    @staticmethod
    def _tokenize_exif(data, app1_address, file_path):
        """
        Tokenizes EXIF data from an APP1 segment.

        Parameters:
            data (bytes): Binary JPEG data.
            app1_address (int): Address of the APP1 segment.
            file_path (str): Path to the JPEG file.

        Returns:
            DataFrame: A DataFrame with all extracted Exif markers.
        """
        endian_marker = data[app1_address + 10:app1_address + 12]
        if endian_marker == b'II':
            endianess = 'little'
        elif endian_marker == b'MM':
            endianess = 'big'
        else:
            raise ValueError(f'Endianess not recognized:\t{endian_marker}')

        # Todo only true if ifd0 starts right here, CIPA DC-008-2012 page 19
        tiff_address = app1_address + 18

        data_0, number_of_fields_0, ifd1_off = parse_tiff.extract_IFD(data[tiff_address:], endianess)
        if ifd1_off == 0:
            return data_0

        ifd1_address = ifd1_off + tiff_address
        try:
            data_1, number_of_fields_1, _ = parse_tiff.extract_IFD(data[ifd1_address:], endianess)
        except ValueError as ve:
            logging.error(f'{ve} @ IFD 1 ({ifd1_address}), IFD0 @ ({tiff_address}) for\n\t{file_path}')
            return data_0

        df = pd.concat([data_0, data_1])
        return df

    @staticmethod
    def _tokenize_xmp(data, app1_address, app1_length):
        """
        NOTE: not used for paper!
        Tokenizes XMP data from an APP1 segment.

        Parameters:
            data (bytes): Binary JPEG data.
            app1_address (int): Address of the APP1 segment.
            file_path (str): Path to the JPEG file.

        Returns:
            DataFrame: A DataFrame with all extracted XMP attributes coded by CRC16.
        """
        # NOTE: we only use attributes
        logging.debug(f'[JPEGAppHasher] Read XML from APP1 @ {app1_address} of {app1_length} Bytes.')
        xmp_meta_off = data[app1_address:].find(b'<x:xmpmeta')
        if xmp_meta_off == -1:
            xmp_meta_off = data[app1_address:].find(b'<?xpacket begin="?" id="W5M0MpCehiHzreSzNTczkc9d"?>')
            if xmp_meta_off == -1:
                logging.warning(
                    f'[JPEGAppHasher] No proper XMP start:\n\t{data[app1_address + 4:app1_address + app1_length + 1]}')
                return []

        xmp_off = app1_address + xmp_meta_off
        xml_data = data[xmp_off:xmp_off + app1_length]

        logging.debug(f'[JPEGAppHasher] Read XML:\n{xml_data}')
        try:
            xml_data = xml_data.decode(encoding='utf-8')
        except UnicodeDecodeError:
            xml_data = xml_data.decode(encoding='ascii', errors='ignore')

        def _iterate_xml(element):
            if element is None:
                logging.warning(f'[JPEGAppHasher] Read XML has no elements:\n{xml_data}')
                return []

            # Todo? include namespace e.g. '{http://ns.oneplus.com/media/1.0}CaptureMode'
            attributes = []
            logging.debug(f'[JPEGAppHasher] XML element:\t{element}')
            for sub_element in element:
                attributes.extend(_iterate_xml(sub_element))
            attributes.extend(element.attrib.keys())
            return attributes

        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(xml_data, parser)
        logging.debug(f'[JPEGAppHasher] XML root:\t{root}')
        xmp_attributes = _iterate_xml(root)
        # print(xmp_attributes)
        xmp_tokens = [crc16.xmodem(attr.encode()).to_bytes(2, 'big') for attr in xmp_attributes]
        # print(xmp_tokens)
        return xmp_tokens

    @staticmethod
    def _tokenize_app1(data, app1_address, file_path):
        """
        Tokenizes APP1 segment.

        Parameters:
            data (bytes): Binary JPEG data.
            app1_address (int): Address of the APP1 segment.
            file_path (str): Path to the JPEG file.

        Returns:
            DataFrame: A DataFrame with all extracted APP1 markers. Only Exif right now.
        """
        """
        NOTE: this part includes the parsing of XMP metadata which is not used for the paper:
        app1_length = int.from_bytes(data[app1_address + 2:app1_address + 4], byteorder="big")
        elif data[app1_address + 4:].startswith(b'http://ns.adobe.com'):
            xmp_tokens: list = JPEGAppHasher._tokenize_xmp(data, app1_address, app1_length)
            return xmp_tokens
        """
        if data[app1_address + 4:app1_address + 8] == b'Exif':
            exif_tokens: pd.DataFrame = JPEGAppHasher._tokenize_exif(data, app1_address, file_path)
            return exif_tokens['value'].tolist()
        else:
            logging.warning(f'Unknown APP1 type!\t{data[app1_address + 4:app1_address + 50]}')
            return []
