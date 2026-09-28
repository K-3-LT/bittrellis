import numpy as np
import pytest

from bittrellis.quant.formats import dequantize_nvfp4, quantize_nvfp4
from bittrellis.quantizers.mse126 import quantize_mse_nvfp4


def test_block_loss_and_byte_replay():
    w = np.random.default_rng(74).normal(size=(9, 64)).astype(np.float32)
    a = quantize_mse_nvfp4(w)
    b = quantize_mse_nvfp4(w)
    assert all(x.tobytes() == y.tobytes() for x, y in zip(a, b, strict=True))
    def losses(result):
        return np.sum((w.astype(np.float64) - dequantize_nvfp4(*result)).reshape(9, 4, 16) ** 2, axis=-1)
    assert np.all(losses(a) <= losses(quantize_nvfp4(w)) + 1e-12)
    assert np.any(losses(a) < losses(quantize_nvfp4(w)) - 1e-8)


def test_chunking_uses_tensor_amax():
    w = np.random.default_rng(92).normal(size=(6, 32)).astype(np.float32)
    amax = float(np.abs(w).max())
    full = quantize_mse_nvfp4(w, amax)
    parts = [quantize_mse_nvfp4(w[i:i + 2], amax) for i in range(0, 6, 2)]
    for index in (0, 1):
        assert np.array_equal(full[index], np.concatenate([p[index] for p in parts]))
    assert all(p[2] == full[2] for p in parts)


def test_zero_weights():
    result = quantize_mse_nvfp4(np.zeros((3, 16), np.float32))
    assert not dequantize_nvfp4(*result).any()


@pytest.mark.parametrize('w', [np.ones((2, 17)), np.empty((0, 16)), np.ones(16), np.full((1, 16), np.nan)])
def test_reject_invalid_weights(w):
    with pytest.raises(ValueError):
        quantize_mse_nvfp4(w)


@pytest.mark.parametrize('amax', [0.5, np.inf, np.nan])
def test_reject_invalid_tensor_scale(amax):
    with pytest.raises(ValueError):
        quantize_mse_nvfp4(np.ones((2, 16)), amax)


def test_complete_tiny_build_and_audit(tiny_all, tmp_path):
    from bittrellis.build import build
    from bittrellis.manifest import Manifest
    from bittrellis.safetensors_io import SafeTensorsDir
    from bittrellis.track import load_track
    from bittrellis.validate import audit

    manifest = Manifest.from_dict({
        'schema': 'bittrellis/manifest@2', 'track': 'HPC-01', 'name': 'test-mse126',
        'default': 'NVFP4',
        'rules': [{'match': 'L*.mlp', 'format': 'NVFP4', 'quantizer': 'mse126'}],
    })
    sources = dict(zip(('base', 'gittensor_nvfp4', 'unsloth_nvfp4'), tiny_all, strict=True))
    first, second = tmp_path / 'first', tmp_path / 'second'
    kwargs = {'verify': False, 'log': lambda *_: None}
    record = build(manifest, load_track('HPC-01'), sources, first, **kwargs)
    repeated = build(manifest, load_track('HPC-01'), sources, second, **kwargs)
    assert record['files'] == repeated['files']
    result = audit(first, manifest, sources, verify_sources=False)
    assert result.ok, result.errors
    assert result.lineage['mse126@v1']['replay_mode'] == 'independent'
    with SafeTensorsDir(first) as built, SafeTensorsDir(tiny_all[1]) as baseline:
        def outside_mlp(name):
            return '.language_model.layers.' not in name or '.mlp.' not in name
        assert {n for n in built.tensors if outside_mlp(n)} == {n for n in baseline.tensors if outside_mlp(n)}
        for name in baseline.tensors:
            if '.language_model.layers.' not in name or '.mlp.' not in name:
                assert bytes(built.raw(name)) == bytes(baseline.raw(name)), name
        name = next(n for n in built.tensors if '.mlp.' in n and n.endswith('.weight'))
        tensor = built.get(name)
    with open(tensor.file, 'r+b') as file:
        file.seek(tensor.offset)
        original = file.read(1)
        file.seek(tensor.offset)
        file.write(bytes([original[0] ^ 1]))
    assert not audit(first, manifest, sources, verify_sources=False).ok


def test_reject_non_mlp_assignment():
    from bittrellis.manifest import Manifest, ManifestError
    from bittrellis.model.qwen38 import Qwen38Arch

    manifest = Manifest.from_dict({
        'schema': 'bittrellis/manifest@2', 'track': 'HPC-01', 'name': 'invalid-scope',
        'default': 'NVFP4',
        'rules': [{'match': 'L*.gdn.*', 'format': 'NVFP4', 'quantizer': 'mse126'}],
    })
    with pytest.raises(ManifestError):
        manifest.expand_assignments(Qwen38Arch().units())
