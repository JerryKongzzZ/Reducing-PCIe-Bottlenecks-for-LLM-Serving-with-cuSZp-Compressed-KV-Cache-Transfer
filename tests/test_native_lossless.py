import os

import pytest

from integration.compression_pipeline.native_lossless import compress, decompress


@pytest.mark.parametrize("codec", ["zstd", "lz4"])
@pytest.mark.parametrize("payload", [b"A" * 65536, os.urandom(65536)])
def test_native_lossless_round_trip(codec, payload):
    encoded = compress(codec, payload)
    assert decompress(codec, encoded, len(payload)) == payload


@pytest.mark.parametrize("codec", ["zstd", "lz4"])
def test_native_lossless_rejects_wrong_output_size(codec):
    payload = b"KV-cache" * 1024
    encoded = compress(codec, payload)
    with pytest.raises(RuntimeError):
        decompress(codec, encoded, len(payload) - 1)
