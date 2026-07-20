"""Small ctypes bindings for the system zstd and LZ4 shared libraries."""

from __future__ import annotations

import ctypes
import ctypes.util


class NativeCodecUnavailable(RuntimeError):
    pass


def _load_library(name: str) -> ctypes.CDLL:
    path = ctypes.util.find_library(name)
    if not path:
        raise NativeCodecUnavailable(f"system library {name!r} is unavailable")
    return ctypes.CDLL(path)


_zstd: ctypes.CDLL | None = None
_lz4: ctypes.CDLL | None = None


def _get_zstd() -> ctypes.CDLL:
    global _zstd
    if _zstd is None:
        library = _load_library("zstd")
        library.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
        library.ZSTD_compressBound.restype = ctypes.c_size_t
        library.ZSTD_compress.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        library.ZSTD_compress.restype = ctypes.c_size_t
        library.ZSTD_decompress.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        library.ZSTD_decompress.restype = ctypes.c_size_t
        library.ZSTD_isError.argtypes = [ctypes.c_size_t]
        library.ZSTD_isError.restype = ctypes.c_uint
        library.ZSTD_getErrorName.argtypes = [ctypes.c_size_t]
        library.ZSTD_getErrorName.restype = ctypes.c_char_p
        _zstd = library
    return _zstd


def _get_lz4() -> ctypes.CDLL:
    global _lz4
    if _lz4 is None:
        library = _load_library("lz4")
        library.LZ4_compressBound.argtypes = [ctypes.c_int]
        library.LZ4_compressBound.restype = ctypes.c_int
        library.LZ4_compress_default.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        library.LZ4_compress_default.restype = ctypes.c_int
        library.LZ4_decompress_safe.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        library.LZ4_decompress_safe.restype = ctypes.c_int
        _lz4 = library
    return _lz4


def _source_buffer(data: bytes) -> ctypes.Array:
    return ctypes.create_string_buffer(data, len(data))


def _zstd_check(library: ctypes.CDLL, code: int) -> int:
    if library.ZSTD_isError(code):
        message = library.ZSTD_getErrorName(code).decode(
            "utf-8", errors="replace"
        )
        raise RuntimeError(f"zstd failure: {message}")
    return int(code)


def compress(codec: str, data: bytes) -> bytes:
    if not data:
        return b""
    source = _source_buffer(data)
    if codec == "zstd":
        library = _get_zstd()
        capacity = int(library.ZSTD_compressBound(len(data)))
        destination = ctypes.create_string_buffer(capacity)
        size = _zstd_check(
            library,
            library.ZSTD_compress(destination, capacity, source, len(data), 1),
        )
    elif codec == "lz4":
        library = _get_lz4()
        if len(data) > 0x7E000000:
            raise ValueError("LZ4 input exceeds the supported single-block size")
        capacity = int(library.LZ4_compressBound(len(data)))
        if capacity <= 0:
            raise RuntimeError("LZ4 could not determine a compression bound")
        destination = ctypes.create_string_buffer(capacity)
        size = int(
            library.LZ4_compress_default(
                source, destination, len(data), capacity
            )
        )
        if size <= 0:
            raise RuntimeError("LZ4 compression failed")
    else:
        raise ValueError(f"unsupported native lossless codec: {codec}")
    return destination.raw[:size]


def decompress(codec: str, data: bytes, original_size: int) -> bytes:
    if original_size < 0:
        raise ValueError("original_size must be non-negative")
    if original_size == 0:
        if data:
            raise RuntimeError("non-empty payload for an empty output")
        return b""
    source = _source_buffer(data)
    destination = ctypes.create_string_buffer(original_size)
    if codec == "zstd":
        library = _get_zstd()
        size = _zstd_check(
            library,
            library.ZSTD_decompress(
                destination, original_size, source, len(data)
            )
        )
    elif codec == "lz4":
        library = _get_lz4()
        size = int(
            library.LZ4_decompress_safe(
                source, destination, len(data), original_size
            )
        )
        if size < 0:
            raise RuntimeError("LZ4 decompression failed")
    else:
        raise ValueError(f"unsupported native lossless codec: {codec}")
    if size != original_size:
        raise RuntimeError(
            f"{codec} produced {size} bytes; expected {original_size}"
        )
    return destination.raw[:size]
