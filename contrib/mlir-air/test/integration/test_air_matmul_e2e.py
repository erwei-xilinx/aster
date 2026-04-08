"""E2E matmul test exercising the real AIR lowering path.

Pipeline:
  mlir-air-opt (preprocess):
    --transform-interpreter
    --air-par-to-herd (forall → herd, each tile = 1 wavefront)
    --one-shot-bufferize
    --air-par-to-launch (outer parallel → launch)
    --air-copy-to-dma (memref.copy → air.dma_memcpy_nd)
    --air-to-amdgcn (flatten hierarchy, herd → wavefront index)
    --convert-memspace-to-amdgcn (integer memspace → #amdgcn.addr_space)
    --convert-to-amdgcn-library-calls (air.dma_memcpy_nd + linalg ops → library calls)
  then aster pipeline:
    --preload → inline → mlir-air-to-asm
"""

import os
import shutil
import subprocess

import numpy as np
import pytest

from aster.execution.helpers import compile_and_run

MCPU = "gfx942"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MLIR_FILE = os.path.join(_THIS_DIR, "..", "air-to-amdgcn-matmul.mlir")
_TRANSFORM_FILE = os.path.join(_THIS_DIR, "..", "air-to-amdgcn-matmul-transform.mlir")
_PADDED_MLIR_FILE = os.path.join(_THIS_DIR, "..", "air-to-amdgcn-matmul-padded.mlir")
_PADDED_TRANSFORM_FILE = os.path.join(
    _THIS_DIR, "..", "air-to-amdgcn-matmul-padded-transform.mlir"
)
_LIBRARY_DIR = os.path.join(
    _THIS_DIR, "..", "..", "..", "..", "mlir_kernels", "library"
)
_KITTENS_DIR = os.path.join(
    _THIS_DIR, "..", "..", "..", "..", "contrib", "kittens", "library"
)

_LIBRARY_PATHS = [
    os.path.join(_LIBRARY_DIR, "common", f)
    for f in [
        "register-init.mlir",
        "indexing.mlir",
        "indexing_ptr.mlir",
        "futures.mlir",
    ]
] + [
    os.path.join(_KITTENS_DIR, f)
    for f in [
        "compute_16x16_f16.mlir",
        "global_16x64_b.mlir",
        "lds_16x64_b.mlir",
        "lds_mfma_16x64_b.mlir",
    ]
]


def _find_mlir_air_opt():
    """Find the mlir-air-opt binary."""
    build_path = os.path.join(
        _THIS_DIR, "..", "..", "..", "..", "build", "bin", "mlir-air-opt"
    )
    if os.path.isfile(build_path):
        return os.path.abspath(build_path)
    path = shutil.which("mlir-air-opt")
    if path:
        return path
    pytest.skip("mlir-air-opt not found")


def _air_preprocess_with_files(mlir_text, transform_file):
    """Run the full AIR lowering pipeline before handing to aster."""
    opt = _find_mlir_air_opt()
    result = subprocess.run(
        [
            opt,
            f"--transform-preload-library=transform-library-paths={transform_file}",
            "--transform-interpreter",
            "--air-par-to-herd",
            "--canonicalize", "--cse",
            "--one-shot-bufferize",
            "--canonicalize", "--cse",
            "--air-par-to-launch=has-air-segment=true",
            "--canonicalize", "--cse",
            "--air-copy-to-dma",
            "--air-to-amdgcn",
            "--canonicalize",
            "--convert-memspace-to-amdgcn",
            "--convert-to-amdgcn-library-calls",
        ],
        input=mlir_text,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"mlir-air-opt AIR preprocessing failed:\n{result.stderr}")
    return result.stdout


def _air_preprocess(mlir_text):
    return _air_preprocess_with_files(mlir_text, _TRANSFORM_FILE)


def _post_air_pipeline(library_paths):
    libs = ",".join(library_paths)
    return (
        "builtin.module("
        "canonicalize,"
        f"amdgcn-preload-library{{library-paths={libs}}},"
        "inline, symbol-dce, canonicalize,"
        "mlir-air-to-asm)"
    )


class TestAirMatmulE2E:

    def test_matmul_64x64(self):
        M, N, K = 64, 64, 64
        np.random.seed(42)
        A = (np.random.randn(M, K) * 0.1).astype(np.float16)
        B_KxN = (np.random.randn(K, N) * 0.1).astype(np.float16)
        B_T = np.ascontiguousarray(B_KxN.T)
        # C must be zero-initialized (fill is erased by convert-to-amdgcn-library-calls;
        # the library's zero_C handles accumulator init per tile).
        C = np.zeros(M * N, dtype=np.float32)

        compile_and_run(
            file_name=_MLIR_FILE,
            kernel_name="matmul_f16_64x64",
            input_data=[A.flatten(), B_T.flatten()],
            output_data=[C],
            pass_pipeline=_post_air_pipeline(_LIBRARY_PATHS),
            library_paths=[],
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),  # 2 wavefronts (2x1 herd)
            preprocess=_air_preprocess,
        )

        expected = (A.astype(np.float32) @ B_KxN.astype(np.float32)).flatten()
        np.testing.assert_allclose(C, expected, rtol=1e-2, atol=1e-2)

    def test_matmul_padded_40x40(self):
        """Matmul with non-tile-aligned dimensions: actual 40x40x64.

        Kernel operates on actual dimensions (memref<40x64xf16>, memref<40x40xf32>).
        Transform pads boundary tiles to 16. Host over-allocates C to 48*48
        to accommodate OOB stores from boundary tiles.
        """
        M, N, K = 40, 40, 64
        M_pad = 48  # next multiple of 16, for C over-allocation

        np.random.seed(42)
        A = (np.random.randn(M, K) * 0.1).astype(np.float16)
        B_T = (np.random.randn(N, K) * 0.1).astype(np.float16)

        # Over-allocate C: kernel writes full 16x16 tiles at boundaries,
        # going beyond 40x40. Allocate 48*48 elements but pass as 40x40 memref.
        C = np.zeros(M_pad * M_pad, dtype=np.float32)

        def padded_preprocess(mlir_text):
            opt = _find_mlir_air_opt()
            result = subprocess.run(
                [
                    opt,
                    f"--transform-preload-library=transform-library-paths={_PADDED_TRANSFORM_FILE}",
                    "--transform-interpreter",
                    "--canonicalize", "--cse",
                    # Set memory_space=2 on padding allocs (no memory space → L1).
                    "--air-override-memref-memory-space=scope=func memory-space=2",
                    "--air-par-to-herd",
                    "--canonicalize", "--cse",
                    "--air-par-to-launch=has-air-segment=true",
                    "--canonicalize", "--cse",
                    "--air-copy-to-dma",
                    "--air-to-amdgcn",
                    "--canonicalize",
                    "--convert-memspace-to-amdgcn",
                    "--convert-to-amdgcn-library-calls",
                ],
                input=mlir_text,
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"mlir-air-opt padded preprocessing failed:\n{result.stderr}")
            return result.stdout

        compile_and_run(
            file_name=_PADDED_MLIR_FILE,
            kernel_name="matmul_f16_40x40",
            input_data=[A.flatten(), B_T.flatten()],
            output_data=[C],
            pass_pipeline=_post_air_pipeline(_LIBRARY_PATHS),
            library_paths=[],
            grid_dim=(1, 1, 1),
            block_dim=(192, 1, 1),  # 3 wavefronts (3x1 herd)
            preprocess=padded_preprocess,
        )

        # Extract valid 40x40 region (C is over-allocated as flat 48*48).
        # The kernel writes with stride=40, so reinterpret accordingly.
        C_2d = C[:M * M_pad].reshape(-1, M_pad)[:M, :N].flatten()
        expected = (A.astype(np.float32) @ B_T.T.astype(np.float32)).flatten()
        np.testing.assert_allclose(C_2d, expected, rtol=1e-2, atol=1e-2)
