# Copyright (C) 2022 Xilinx, Inc
#
# SPDX-License-Identifier: BSD-3-Clause

from enum import Enum
import numpy as np
import json
import os
from pynq import DefaultIP
from pynq.ps import CPU_ARCH, ZU_ARCH
import struct

__author__ = "Mario Ruiz"
__copyright__ = "Copyright 2022, Xilinx"
__email__ = "pynq_support@xilinx.com"


def _float2int(value: float) -> int:
    """Pack a single precision floating point into a 32-bit integer"""
    return int.from_bytes(struct.pack('f', np.single(value)), 'little')


if CPU_ARCH == ZU_ARCH:
    _cols = 1920
    _rows = 1080
else:
    _cols = 1280
    _rows = 720


class VitisVisionIP(DefaultIP):
    """Generic Driver for Vitis Vision IP cores"""

    bindto = [
        'xilinx.com:hls:dma2video_accel:1.0',
        'xilinx.com:hls:video2dma_accel:1.0',
        'xilinx.com:hls:rgb2gray_accel:1.0',
        'xilinx.com:hls:medianBlur_accel:1.0',
        'xilinx.com:hls:gray2rgb_accel:1.0',
        'xilinx.com:hls:pyrUp_accel:1.0',
        'xilinx.com:hls:subtract_accel:1.0',
        'xilinx.com:hls:rgb2hsv_accel:1.0',
        'xilinx.com:hls:rgb2xyz_accel:1.0',
        "xilinx.com:hls:absdiff_accel:1.0",
        "xilinx.com:hls:add_accel:1.0",
        "xilinx.com:hls:bitwise_and_accel:1.0",
        "xilinx.com:hls:bitwise_not_accel:1.0",
        "xilinx.com:hls:bitwise_or_accel:1.0",
        "xilinx.com:hls:bitwise_xor_accel:1.0",
    ]

    _rows_offset = 0x10
    _cols_offset = 0x18

    def __init__(self, description):
        super().__init__(description=description)

    def start(self):
        """Populate the image resolution and start the IP"""
        file = "/tmp/resolution.json"
        if os.path.exists(file):
            with open(file, "r", encoding='utf8') as f:
                reso = json.load(f)
                self._cols, self._rows = reso["width"], reso["height"]
        else:
            self._cols, self._rows = _cols, _rows
        self.write(self._rows_offset, int(self._rows))
        self.write(self._cols_offset, int(self._cols))
        self.write(0x00, 0x81)

    def stop(self):
        """Stop the IP"""
        self.write(0x00, 0x0)

    def status(self):
        return self.register_map.CTRL

    @property
    def rows(self) -> int:
        """Image height"""
        rows = int(self.read(self._rows_offset))
        return rows

    @rows.setter
    def rows(self, rows: int):
        if not isinstance(rows, int):
            raise ValueError("rows must an integer")
        elif rows < 0:
            raise ValueError("rows cannot be negative")

        self._rows = rows
        self.write(self._rows_offset, int(self._rows))

    @property
    def cols(self) -> int:
        """Image width"""
        cols = int(self.read(self._cols_offset))
        return cols

    @cols.setter
    def cols(self, cols: int):
        if not isinstance(cols, int):
            raise ValueError("cols must an integer")
        elif cols < 0:
            raise ValueError("cols cannot be negative")

        self._cols = cols
        self.write(self._cols_offset, int(self._cols))


class XvF2d(Enum):
    """Supported filter2D kernels"""
    identity = 0
    edge_x = 1
    edge_y = 2
    edge = 3
    sharpen = 4
    sobel_x = 5
    sobel_y = 6
    scharr_x = 7
    scharr_y = 8
    prewitt_x = 9
    prewitt_y = 10
    gaussian_blur = 11
    median_blur = 12


class Filter2d(VitisVisionIP):
    """Filter 2D Kernel"""

    bindto = ['xilinx.com:hls:filter2d_accel:1.0']
    _size = 3

    def __init__(self, description):
        super().__init__(description=description)
        self._kernel = np.zeros((self._size, self._size), dtype=np.int16)
        self._kernel[(self._size // 2) + 1][(self._size // 2) + 1] = 1
        self._quantize_error = 0
        self._shift = 0

        self._kernel_type = XvF2d.identity
        self._sigma = 1.0

    def _gaussianBlur(self):
        """Compute a Gaussian kernel of a given size and sigma.
        Implementation based on
        """
        kernel = np.zeros((self._size, self._size), dtype=float)
        for u in range(kernel.shape[0]):
            for v in range(kernel.shape[1]):
                uc = u - (kernel.shape[0] - 1) / 2
                vc = v - (kernel.shape[1] - 1) / 2
                g = np.exp(
                    -(np.power(uc, 2) + np.power(vc, 2))
                    / (2 * np.power(self._sigma, 2))
                )
                kernel[u][v] = g

        return kernel / np.sum(kernel)

    def _medianBlur(self):
        kernel = np.ones((self._size, self._size), dtype=float)
        return kernel / np.sum(kernel)

    def _quantiseKernel(self, kernel, bit_width: int = 16,
                        max_shift: int = 255):
        """Quantise the floating point kernel into integer taking into account
        the maximum element in the kernel
        """
        max_value = np.max(kernel)
        scaling_max = (np.power(2, bit_width - 1)) / max_value
        shift_up = int(np.floor(np.log2(scaling_max)))
        scale_factor = np.power(2, shift_up) - 1
        kernel_q = np.rint(kernel * scale_factor)
        self._quantize_error = (kernel * scale_factor) - kernel_q

        return kernel_q.astype(np.int16), shift_up

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        if not isinstance(sigma, (float, int)):
            raise ValueError("sigma must a number")

        self._sigma = float(sigma)

        if self._kernel_type == XvF2d.gaussian_blur:
            self._kernel, self._shift = \
                self._quantiseKernel(self._gaussianBlur())
            self._populateKernel()

    @property
    def kernel_type(self):
        return self._kernel_type

    @kernel_type.setter
    def kernel_type(self, kernel_type: XvF2d):
        if kernel_type not in XvF2d:
            raise ValueError("Kernel type unknown")

        self._shift = 0
        if kernel_type == XvF2d.identity:
            self._kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                    dtype=np.int16)
        elif kernel_type == XvF2d.edge_x:
            self._kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                                    dtype=np.int16)
        elif kernel_type == XvF2d.edge_y:
            self._kernel = np.array([[1, 0, -1], [0, 4, 0], [-1, 0, 1]],
                                    dtype=np.int16)
        elif kernel_type == XvF2d.edge:
            self._kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
                                    dtype=np.int16)
        elif kernel_type == XvF2d.sobel_x:
            self._kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                    dtype=np.int16)
        elif kernel_type == XvF2d.sobel_y:
            self._kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                    dtype=np.int16)
        elif kernel_type == XvF2d.sharpen:
            self._kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
                                    dtype=np.int16)
        elif kernel_type == XvF2d.scharr_x:
            self._kernel = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]],
                                    dtype=np.int16)
        elif kernel_type == XvF2d.scharr_y:
            self._kernel = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]],
                                    dtype=np.int16)
        elif kernel_type == XvF2d.prewitt_x:
            self._kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]],
                                    dtype=np.int16)
        elif kernel_type == XvF2d.prewitt_y:
            self._kernel = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
                                    dtype=np.int16)
        elif kernel_type == XvF2d.median_blur:
            self._kernel, self._shift = \
                self._quantiseKernel(self._medianBlur())
            self._shift -= 1
        elif kernel_type == XvF2d.gaussian_blur:
            self._kernel, self._shift = \
                self._quantiseKernel(self._gaussianBlur())

        self._kernel_type = kernel_type
        self._populateKernel()

    def _populateKernel(self):
        kernel = \
            self._kernel.reshape(self._kernel.shape[0] * self._kernel.shape[1])
        aux = 0
        populate = False
        for i in range(len(kernel)):
            if i % 2 == 0:
                aux = kernel[i]
            else:
                aux = ((np.uint32(kernel[i]) << 16) & 0xFFFF0000) + aux
                populate = True

            if populate or (i == len(kernel) - 1):
                self.write(0x40 + ((i // 2) * 4), int(aux))
                aux = 0
                populate = False

        self.write(0x20, int(self._shift))

    def start(self):
        super().start()
        self._populateKernel()


class DuplicateIP(VitisVisionIP):
    """DuplicateIP driver"""

    bindto = ['xilinx.com:hls:duplicate_accel:1.0']

    _rows_offset = 0x1EC
    _cols_offset = 0x1F4


class GaussianBlur(VitisVisionIP):
    """GaussianBlur"""

    bindto = ['xilinx.com:hls:GaussianBlur_accel:1.0']

    def __init__(self, description):
        super().__init__(description=description)
        self.sigma = 1.0

    def start(self):
        super().start()
        if self.sigma < 0.27:
            aux = 0.27
        else:
            aux = self.sigma
        self.write(0x20, _float2int(aux))


class ColorThreshold(VitisVisionIP):
    """Color Thresholding IP driver
    lower_thr and upper_thr are a numpy array, each row corresponds to a
    channel in the pixel.
    For RGB, row 0 is R, row 1 is G and row 2 is B
    For HSV, row 0 is H, row 1 is S and row 2 is V
    For XYZ, row 0 is X, row 1 is Y and row 2 is Z

    """

    bindto = ['xilinx.com:hls:colorthresholding_accel:1.0']

    def __init__(self, description):
        super().__init__(description=description)
        self._lower_thr = np.array(
            [[22, 38, 160], [150, 150, 150], [60, 60, 60]], dtype=np.uint8
        )
        self._upper_thr = np.array(
            [[38, 75, 179], [255, 255, 255], [255, 255, 255]], dtype=np.uint8
        )

    def _populateThreshold(self):
        lower = self._lower_thr.reshape(
            self._lower_thr.shape[0] * self._lower_thr.shape[1]
        )
        upper = self._upper_thr.reshape(
            self._upper_thr.shape[0] * self._upper_thr.shape[1]
        )
        aux = 0
        for i in range(lower.shape[0]):
            shift = (i % 4) * 8
            aux = (np.uint32(lower[i]) << shift) + aux

            if ((i + 1) % 4) == 0 or i == (len(lower) - 1):
                self.write(0x20 + (i // 4) * 4, int(aux))
                aux = 0
        aux = 0
        for i in range(upper.shape[0]):
            shift = (i % 4) * 8
            aux = (np.uint32(upper[i]) << shift) + aux

            if ((i + 1) % 4) == 0 or i == (len(upper) - 1):
                self.write(0x60 + (i // 4) * 4, int(aux))
                aux = 0

    def start(self):
        super().start()
        self._populateThreshold()

    @property
    def lower_thr(self):
        """Set and retrieve lower threshold configuration"""
        return self._lower_thr

    @lower_thr.setter
    def lower_thr(self, threshold):
        if not isinstance(threshold, np.ndarray):
            raise ValueError("lower_thr expects a numpy ndarray as input")
        elif threshold.shape != self._lower_thr.shape:
            raise ValueError(
                "Shapes do not match, lower_thr expects a {} ndarray".format(
                    self._lower_thr.shape
                )
            )
        self._lower_thr = threshold
        self._populateThreshold()

    @property
    def upper_thr(self):
        """Set and retrieve lower threshold configuration"""
        return self._upper_thr

    @upper_thr.setter
    def upper_thr(self, threshold):
        if not isinstance(threshold, np.ndarray):
            raise ValueError("upper_thr expects a numpy ndarray as input")
        elif threshold.shape != self._upper_thr.shape:
            raise ValueError(
                "Shapes do not match, upper_thr expects a {} ndarray".format(
                    self._upper_thr.shape
                )
            )
        self._upper_thr = threshold
        self._populateThreshold()


class InRange(VitisVisionIP):
    """inRange"""

    bindto = ['xilinx.com:hls:inRange_accel:1.0']

    def __init__(self, description):
        super().__init__(description=description)
        self.lower_thr = np.array([22, 150, 60], dtype=np.uint8)
        self.upper_thr = np.array([38, 255, 255], dtype=np.uint8)

    def populateThreshold(self):
        lower = self.lower_thr
        upper = self.upper_thr
        aux = 0
        for i in range(lower.shape[0]):
            shift = (i % 4) * 8
            aux = (np.uint32(lower[i]) << shift) + aux

            if ((i + 1) % 4) == 0 or i == (len(lower) - 1):
                self.write(0x20 + (i // 4) * 4, int(aux))
                aux = 0
        aux = 0
        for i in range(upper.shape[0]):
            shift = (i % 4) * 8
            aux = (np.uint32(upper[i]) << shift) + aux

            if ((i + 1) % 4) == 0 or i == (len(upper) - 1):
                self.write(0x30 + (i // 4) * 4, int(aux))
                aux = 0

    def start(self):
        super().start()
        self.populateThreshold()


class Morphological(VitisVisionIP):
    """Erode and dilate"""

    bindto = [
        'xilinx.com:hls:dilate_accel:1.0',
        'xilinx.com:hls:erode_accel:1.0'
        ]

    def __init__(self, description):
        super().__init__(description=description)
        self.kernel = np.ones((3, 3), dtype=np.uint8)

    def populateKernel(self):
        kernel = \
            self.kernel.reshape(self.kernel.shape[0] * self.kernel.shape[1])
        aux = 0
        for i in range(len(kernel)):
            shift = (i % 4) * 8
            aux = (kernel[i] << shift) + aux

            if ((i + 1) % 4) == 0 or (i == len(kernel) - 1):
                self.write(0x40 + ((i // 4) * 4), int(aux))
                aux = 0

    def start(self):
        super().start()
        self.populateKernel()


class Fast(VitisVisionIP):
    """Corner Detect, using Fast algorithm IP, python driver"""

    bindto = ['xilinx.com:hls:fast_accel:1.0']
    _max_threshold = (2 ** 8) - 1

    def __init__(self, description):
        super().__init__(description=description)
        self._threshold = 20

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        if not isinstance(threshold, int):
            raise ValueError("threshold must be int")
        elif threshold > self._max_threshold:
            raise ValueError("threshold cannot be bigger than {}"
                             .format(self._max_threshold))
        self._threshold = threshold
        self.write(0x20, int(self._threshold))

    def start(self):
        super().start()
        self.write(0x20, int(self._threshold))


def _convert_to_q0_16(v, maxval=(2 ** 16) - 1):
    vtmp = int(v * (2 ** 16))
    return vtmp if vtmp <= maxval else maxval


class CornerHarris(VitisVisionIP):
    """Corner Detector, using Harris algorithm IP, python driver"""

    bindto = ['xilinx.com:hls:cornerHarris_accel:1.0']
    _max_threshold = (2 ** 16) - 1

    def __init__(self, description):
        super().__init__(description=description)
        # set default threshold and k as per documentation
        self._threshold = 442
        self._k = _convert_to_q0_16(0.04)

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        if not isinstance(threshold, int):
            raise ValueError("threshold must be int")
        elif threshold > self._max_threshold:
            raise ValueError("threshold cannot be bigger than {}"
                             .format(self._max_threshold))

        self._threshold = threshold
        self.write(0x20, int(self._threshold))

    @property
    def k(self):
        """Harris detector parameter"""
        return self._k

    @k.setter
    def k(self, k):
        if not isinstance(k, float):
            raise ValueError("k must be float")
        elif k > 1.0:
            raise ValueError("k must be between 0.0 and 1.0")

        self._k = _convert_to_q0_16(k)
        self.write(0x28, int(self._k))

    def start(self):
        super().start()
        self.write(0x20, int(self._threshold))
        self.write(0x28, int(self._k))


class XvLut(Enum):
    """Supported LUT kernels"""
    identity = 0
    negative = 1
    binary_threshold = 2
    group_bin = 3
    offset = 4
    threshold = 5
    random = 6


class PixelLut(VitisVisionIP):
    """Lut IP"""

    bindto = ['xilinx.com:hls:LUT_accel:1.0', 'xilinx.com:hls:lut_accel:1.0']

    def __init__(self, description):
        super().__init__(description=description)
        self._lut = np.empty((3, 256), dtype=np.uint8)
        self.step = 8
        self.offset = 32
        self._shape = self._lut.shape
        self.kernel_type = XvLut.negative
        self._threshold = np.random.randint(0, 255, (2, 3, 3), dtype=np.uint8)

    def _negative(self):
        for c in range(self._shape[0]):
            for e in range(self._shape[1]):
                self._lut[c][e] = 255 - e

    def _identity(self):
        for c in range(self._shape[0]):
            for e in range(self._shape[1]):
                self._lut[c][e] = e

    def _binary_threshold(self):
        for c in range(self._shape[0]):
            for e in range(self._shape[1]):
                if e < 128:
                    self._lut[c][e] = 0
                else:
                    self._lut[c][e] = 255

    def _group_bin(self, step=8):
        for c in range(self._shape[0]):
            for e in range(self._shape[1]):
                self._lut[c][e] = (e // self.step) * self.step

    def _offset(self):
        for c in range(self._shape[0]):
            for e in range(self._shape[1]):
                aux = (int(e) + self.offset) % 256
                self._lut[c][e] = np.uint8(aux)

    def _custom_threshold(self):
        """Fill range between lower and upper threshold with 255 for
        each channel
        """

        lut = np.zeros(self._lut.shape, dtype=np.uint8)

        for c in range(lut.shape[0]):
            min_value = self._threshold[0][c]
            max_value = self._threshold[1][c]
            for idx, e in enumerate(min_value):
                for v in range(min_value[idx], max_value[idx]):
                    lut[2 - c][v] = 255

        self._lut = lut

    @property
    def threshold(self):
        """Set and retrieve threshold ndarray

        The first index indicates

            0: lower threshold
            1: upper threshold

        The second index indicates the channel
        The third index indicates the value
        """

        return self._threshold

    @threshold.setter
    def threshold(self, matrix):
        if not isinstance(matrix, np.ndarray):
            raise ValueError("threshold expects a numpy ndarray as input")
        elif matrix.shape != self._threshold.shape:
            raise ValueError(
                "Shapes do not match, threshold expects a {} ndarray".format(
                    self._threshold.shape
                )
            )

        self._threshold = matrix
        self.kernel_type = XvLut.threshold

    @property
    def kernel_type(self):
        return self._kernel_type

    @kernel_type.setter
    def kernel_type(self, kernel_type: XvLut):
        if kernel_type not in XvLut:
            raise ValueError("Kernel type unknown")

        if kernel_type == XvLut.negative:
            self._negative()
        elif kernel_type == XvLut.identity:
            self._identity()
        elif kernel_type == XvLut.binary_threshold:
            self._binary_threshold()
        elif kernel_type == XvLut.group_bin:
            self._group_bin()
        elif kernel_type == XvLut.offset:
            self._offset()
        elif kernel_type == XvLut.threshold:
            self._custom_threshold()
        elif kernel_type == XvLut.random:
            self._lut = np.random.randint(0, 255, self._shape, dtype=np.uint8)

        self._kernel_type = kernel_type
        self._populateLUT()

    def _populateLUT(self):
        kernel = self._lut.reshape(self._shape[0] * self._shape[1])
        aux = 0
        for i in range(len(kernel)):
            shift = (i % 4) * 8
            aux = (kernel[i] << shift) + aux

            if ((i + 1) % 4) == 0 or (i == len(kernel) - 1):
                self.write(0x400 + ((i // 4) * 4), int(aux))
                aux = 0

    def start(self):
        super().start()
        self._populateLUT()

    @property
    def lut(self):
        return self._lut

    @lut.setter
    def lut(self, lut: np.ndarray):

        if lut.dtype != self._lut.dtype:
            raise TypeError("Wrong type, expect type {}"
                            .format(self._lut.dtype))
        elif lut.shape != self._shape:
            raise TypeError("Wrong shape, expect shape {}".format(self._shape))

        self._lut = lut
        self._populateLUT()


class MultiplyIP(VitisVisionIP):
    """Pixel-wise multiplication"""

    bindto = ['xilinx.com:hls:multiply_accel:1.0']

    def __init__(self, description):
        super().__init__(description=description)
        self._scale = 1.0

    def start(self):
        super().start()
        self.write(0x20, _float2int(self.scale))

    @property
    def scale(self):
        """Scale value

        Each pixel is multiplied by this scale value

        """
        return self._scale

    @scale.setter
    def scale(self, scale):
        if not isinstance(scale, (int, float)):
            raise ValueError("scale should be int or float")
        elif scale < 0:
            raise ValueError("scale cannot be negative")

        self._scale = float(scale)
        self.write(0x20, _float2int(self.scale))

video_mixer_regs = {
    'CTRL':
        {'address_offset': 0, 'size': 32, 'access': 'read-write', 'description': 'Control signals', 'fields': {
            'ap_start': {'bit_offset': 0, 'bit_width': 1, 'description': 'Control signals', 'access': 'read-write'},
            'ap_done': {'bit_offset': 1, 'bit_width': 1, 'description': 'Control signals', 'access': 'read-only'},
            'ap_idle': {'bit_offset': 2, 'bit_width': 1, 'description': 'Control signals', 'access': 'read-only'},
            'ap_ready': {'bit_offset': 3, 'bit_width': 1, 'description': 'Control signals', 'access': 'read-only'},
            'flush_pending': {'bit_offset': 5, 'bit_width': 1, 'description': 'Flush pending AXI transactions', 'access': 'read-write'},
            'flush_done': {'bit_offset': 6, 'bit_width': 1, 'description': 'Flush Done', 'access': 'read-only'},
            'auto_restart': {'bit_offset': 7, 'bit_width': 1, 'description': 'Control signals', 'access': 'read-write'}}
        },
    'GIER': {'address_offset': 4, 'size': 32, 'access': 'read-write', 'description': 'Global Interrupt Enable Register', 'fields': {
        'Enable': {'bit_offset': 0, 'bit_width': 1, 'description': 'Global Interrupt Enable Register', 'access': 'read-write'}}
    },
    'IP_IER': {'address_offset': 8, 'size': 32, 'access': 'read-write', 'description': 'IP Interrupt Enable Register', 'fields': {
        'ap_done': {'bit_offset': 0, 'bit_width': 1, 'description': 'IP Interrupt Enable Register', 'access': 'read-write'},
        'ap_ready': {'bit_offset': 1, 'bit_width': 1, 'description': 'IP Interrupt Enable Register', 'access': 'read-write'}}
    },
    'IP_ISR': {'address_offset': 12, 'size': 32, 'access': 'read-write', 'description': 'IP Interrupt Status Register', 'fields': {
        'ap_done': {'bit_offset': 0, 'bit_width': 1, 'description': 'IP Interrupt Status Register', 'access': 'read-only'},
        'ap_ready': {'bit_offset': 1, 'bit_width': 1, 'description': 'IP Interrupt Status Register', 'access': 'read-only'}}
    },
    'Width': {'address_offset': 16, 'size': 32, 'access': 'read-write', 'description': 'Active width of background'},
    'Height': {'address_offset': 24, 'size': 32, 'access': 'read-write', 'description': 'Active height of background'},
    'background_r_or_y': {'address_offset': 40, 'size': 32, 'access': 'read-write', 'description': 'Red or Y value of background color'},
    'background_u_or_g': {'address_offset': 48, 'size': 32, 'access': 'read-write', 'description': 'Green or U value of background color'},
    'background_g_or_u': {'address_offset': 56, 'size': 32, 'access': 'read-write', 'description': 'Blue or V value of background color'},
    'layer_enabled': {'address_offset': 64, 'size': 32, 'access': 'read-write', 'description': 'Layer enable', 'fields': {
        'master_layer': {'bit_offset': 0, 'bit_width': 1, 'description': 'Master layer is enabled/disabled', 'access': 'read-write'},
        'overlay_layer_1': {'bit_offset': 1, 'bit_width': 1, 'description': 'Overlay Layer 1 is enabled/disabled', 'access': 'read-write'},
        'logo_layer': {'bit_offset': 23, 'bit_width': 1, 'description': 'Logo layer is enabled/disabled', 'access': 'read-write'}
    }},

    'layer_1_alpha': {'address_offset': 512, 'size': 32, 'access': 'read-write', 'description': 'Alpha blending value for layer 1'},
    'layer_1_start_x': {'address_offset': 520, 'size': 32, 'access': 'read-write', 'description': 'X position of the top left corner of layer 1, relative to the background layer'},
    'layer_1_start_y': {'address_offset': 528, 'size': 32, 'access': 'read-write', 'description': 'Y position of the top left corner of layer 1, relative to the background layer'},
    'layer_1_width': {'address_offset': 536, 'size': 32, 'access': 'read-write', 'description': 'Active width (in pixels) of layer 1'},
    'layer_1_stride': {'address_offset': 544, 'size': 32, 'access': 'read-write', 'description': 'Active stride (in bytes) of layer 1'},
    'layer_1_height': {'address_offset': 552, 'size': 32, 'access': 'read-write', 'description': 'Active height  (in lines) of layer 1'},
    'layer_1_scale_factor': {'address_offset': 560, 'size': 32, 'access': 'read-write', 'description': 'Scale factor for layer 1'},
    'layer_1_plane_1_buffer': {'address_offset': 576, 'size': 32, 'access': 'read-write', 'description': 'Start address of plane 1 of frame buffer for layer 1'}
}

from pynq import allocate
import cv2

class VideoMixer(DefaultIP):
    """Video Mixer"""

    bindto = ['xilinx.com:ip:v_mix:5.2']

    def __init__(self, description):
        description['registers'] = video_mixer_regs
        super().__init__(description=description)

    def start(self):
        """Populate the image resolution and start the IP"""
        file = "/tmp/resolution.json"
        if os.path.exists(file):
            with open(file, "r", encoding='utf8') as f:
                reso = json.load(f)
                self._cols, self._rows = reso["width"], reso["height"]
        else:
            self._cols, self._rows = _cols, _rows
        self.write(16, int(self._cols))
        self.write(24, int(self._rows))
        self.register_map.layer_enabled.master_layer = 1
        self.write(0x00, 0x81)

    def overlay_1(self, filename, scale:int=0):
        """This one is almost working"""
        img = cv2.imread(filename)[:,:, [2, 0,1]]
        self.logo = allocate(img.shape, dtype=np.uint8)
        self.logo[:] = img[:]
        self.register_map.layer_1_alpha = 200
        self.register_map.layer_1_start_x = 100
        self.register_map.layer_1_start_y = 200
        self.register_map.layer_1_width = img.shape[1] * (2**scale)
        self.register_map.layer_1_height = img.shape[0] * (2**scale)
        self.register_map.layer_1_stride = img.shape[1] * img.shape[2]
        self.register_map.layer_1_scale_factor = int(scale)
        self.register_map.layer_1_plane_1_buffer = self.logo.physical_address
        self.register_map.layer_enabled.overlay_layer_1 = 1

    def overlay_1_with_extra_channel(self, filename, scale:int=0):
        """This one is working"""
        img = cv2.imread(filename)[:,:, [2, 0,1]]
        rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        self.logo = allocate(rgba.shape, dtype=np.uint8)
        self.logo[:] = rgba[:]
        self.register_map.layer_1_alpha = 200
        self.register_map.layer_1_start_x = 100
        self.register_map.layer_1_start_y = 200
        self.register_map.layer_1_width = rgba.shape[1] * (2**scale)
        self.register_map.layer_1_height = rgba.shape[0] * (2**scale)
        self.register_map.layer_1_stride = rgba.shape[1] * rgba.shape[2]
        self.register_map.layer_1_scale_factor = int(scale)
        self.register_map.layer_1_plane_1_buffer = self.logo.physical_address
        self.register_map.layer_enabled.overlay_layer_1 = 1

    def disable_overlays(self):
        self.register_map.layer_enabled = 0x1
