"""Weight-only NVFP4 block-scale search; no activation or model-quality claim.

Keep the RTN tensor scale and ModelOpt storage layout. Search positive finite
E4M3 block scales by float64 reconstruction loss, with RTN winning ties.
Only the searchable MLP Linear is read; no calibration or external state.
"""
import numpy as np

from ..precision import NVFP4
from ..quant.formats import (
    E2M1_TABLE,
    E4M3_TABLE,
    e2m1_encode,
    pack_nibbles,
    quantize_nvfp4,
    unpack_nibbles,
)
from .base import Quantizer, f32_rows


def quantize_mse_nvfp4(w, global_amax=None):
    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 2 or not w.size or w.shape[1] % 16 or not np.isfinite(w).all():
        raise ValueError('expected finite nonempty [rows, channels divisible by 16]')
    amax = float(np.abs(w).max()) if global_amax is None else float(global_amax)
    if not np.isfinite(amax) or amax < float(np.abs(w).max()):
        raise ValueError('global_amax must be finite and cover the weights')
    packed, scales, ws2 = quantize_nvfp4(w, global_amax=amax)
    if not ws2 > 0:
        raise ValueError('global scale underflow')
    blocks = w.reshape(w.shape[0], -1, 16).astype(np.float64)
    codes = unpack_nibbles(packed).reshape(blocks.shape)
    decoded = E2M1_TABLE[codes] * (E4M3_TABLE[scales] * ws2)[..., None]
    best = np.sum((blocks - decoded) ** 2, axis=-1)
    for code in range(1, 127):
        scale = np.float32(E4M3_TABLE[code] * ws2)
        if scale == 0:
            continue
        q = e2m1_encode((blocks / scale).astype(np.float32))
        decoded = E2M1_TABLE[q] * scale
        loss = np.sum((blocks - decoded) ** 2, axis=-1)
        take = loss < best
        best[take] = loss[take]
        scales[take] = code
        codes[take] = q[take]
    return pack_nibbles(codes.reshape(w.shape)), scales, ws2


class MSE126(Quantizer):
    name = "mse126"
    version = 1
    formats = (NVFP4,)
    lineage = "regenerable"
    replay_mode = "independent"
    description = "NVFP4, 126 legal block scales minimizing weight reconstruction loss"

    def supports(self, unit, fmt):
        return unit.kind == "mlp" and fmt == NVFP4

    def encode(self, ctx, unit, lin, fmt):
        if not self.supports(unit, fmt):
            raise ValueError("mse126 supports MLP NVFP4 only")
        amax = 0.0
        for _, rows in f32_rows(ctx, lin, chunk=64):
            if not np.isfinite(rows).all():
                raise ValueError("base weights must be finite")
            amax = max(amax, float(np.abs(rows).max()))
        packed = np.empty((lin.rows, lin.cols // 2), np.uint8)
        scales = np.empty((lin.rows, lin.cols // 16), np.uint8)
        for start, rows in f32_rows(ctx, lin, chunk=64):
            p, s, ws2 = quantize_mse_nvfp4(rows, global_amax=amax)
            packed[start:start + len(rows)] = p
            scales[start:start + len(rows)] = s
        return [(".weight", "U8", packed.shape, packed),
                (".weight_scale", "F8_E4M3", scales.shape, scales),
                (".weight_scale_2", "F32", (), np.asarray(ws2, "<f4").reshape(()))]
