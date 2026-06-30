#!/usr/bin/env python3
"""Quantize Higgs Audio GGUF weights to mostly-Q4_K.

This script is self-contained for the C++ runtime repository:
- GGUF parsing/writing uses the public `gguf` Python package.
- Q4_K quantization calls this repository's vendored GGML through a tiny
  temporary ctypes helper compiled from `vendor/ggml`.

Only large 2D matrix weights are quantized. Codec convolution/codebook tensors,
norms, biases, and token/output embeddings stay in their original precision.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any

import numpy as np

try:
    from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType
    from gguf.constants import GGML_QUANT_SIZES, LlamaFileType
    from gguf.gguf_writer import GGUFValue, GGUFWriter, TensorInfo
except ImportError as exc:
    raise SystemExit("missing dependency: install with `python -m pip install gguf numpy`") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path(os.environ.get("HIGGS_GGUF", "/root/code/ggbond/models/higgs-audio-v3-tts-4b-mixed.gguf"))
DEFAULT_OUTPUT = Path(os.environ.get("HIGGS_Q4_K_GGUF", "/root/code/ggbond/models/higgs-audio-v3-tts-4b-q4_k.gguf"))
GGML_TYPE_Q4_K = 12

SKIP_NAME_PARTS = (
    "token_embd",
    "position",
    "norm",
    "bias",
    "conv",
    "codebook",
    "quantizer",
    "snake",
    "freq",
    "rope",
)


@dataclass(frozen=True)
class TensorPlan:
    name: str
    shape: tuple[int, ...]
    ne: tuple[int, ...]
    input_type: GGMLQuantizationType
    output_type: GGMLQuantizationType
    input_nbytes: int
    output_nbytes: int
    quantize: bool


class GgmlQuantizer:
    def __init__(self, project_root: Path) -> None:
        self.lib_path = self._build_helper(project_root)
        self.lib = ctypes.CDLL(str(self.lib_path))
        self.lib.higgs_ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_size_t]
        self.lib.higgs_ggml_row_size.restype = ctypes.c_size_t
        self.lib.higgs_ggml_quantize_chunk.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        self.lib.higgs_ggml_quantize_chunk.restype = ctypes.c_size_t

    @staticmethod
    def _build_helper(project_root: Path) -> Path:
        ggml_root = project_root / "vendor/ggml"
        if not (ggml_root / "src/ggml.c").exists():
            raise SystemExit(f"missing GGML sources: {ggml_root}; run `git submodule update --init --recursive`")

        build_dir = Path(tempfile.gettempdir()) / "higgs-audio-ggml-cpp-quantize"
        build_dir.mkdir(parents=True, exist_ok=True)
        src = build_dir / "higgs_ggml_quantize_helper.c"
        so = build_dir / "libhiggs_ggml_quantize_helper.so"
        source = r'''
#include "ggml.h"
#include <stddef.h>
#include <stdint.h>

size_t higgs_ggml_row_size(int type, size_t n_per_row) {
    return ggml_row_size((enum ggml_type) type, n_per_row);
}

size_t higgs_ggml_quantize_chunk(int type, const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return ggml_quantize_chunk((enum ggml_type) type, src, dst, 0, nrows, n_per_row, NULL);
}
'''
        if not src.exists() or src.read_text() != source:
            src.write_text(source)

        c_inputs = [
            src,
            ggml_root / "src/ggml.c",
            ggml_root / "src/ggml-quants.c",
            ggml_root / "src/ggml-alloc.c",
        ]
        cxx_inputs = [
            ggml_root / "src/ggml-backend-dl.cpp",
            ggml_root / "src/ggml-backend.cpp",
            ggml_root / "src/ggml-backend-meta.cpp",
            ggml_root / "src/ggml-backend-reg.cpp",
            ggml_root / "src/ggml-threading.cpp",
        ]
        inputs = c_inputs + cxx_inputs
        if so.exists() and so.stat().st_mtime >= max(path.stat().st_mtime for path in inputs):
            return so

        common_flags = [
            "-O3",
            "-fPIC",
            "-D_GNU_SOURCE",
            "-DM_PI=3.14159265358979323846",
            "-DGGML_VERSION=\"higgs-audio.cpp\"",
            "-DGGML_COMMIT=\"standalone-quantizer\"",
            "-I",
            str(ggml_root / "include"),
            "-I",
            str(ggml_root / "src"),
        ]
        objects: list[Path] = []
        cc = os.environ.get("CC", "cc")
        cxx = os.environ.get("CXX", "c++")
        for input_path in c_inputs:
            obj = build_dir / (input_path.name.replace(".", "_") + ".o")
            subprocess.run([cc, "-std=c11", *common_flags, "-c", str(input_path), "-o", str(obj)], check=True)
            objects.append(obj)
        for input_path in cxx_inputs:
            obj = build_dir / (input_path.name.replace(".", "_") + ".o")
            subprocess.run([cxx, "-std=c++17", *common_flags, "-c", str(input_path), "-o", str(obj)], check=True)
            objects.append(obj)

        cmd = [
            cxx,
            "-std=c++17",
            "-shared",
            *(str(obj) for obj in objects),
            "-pthread",
            "-o",
            str(so),
        ]
        subprocess.run(cmd, check=True)
        return so

    def row_size(self, n_per_row: int) -> int:
        return int(self.lib.higgs_ggml_row_size(GGML_TYPE_Q4_K, n_per_row))

    def q4_k(self, data: np.ndarray[Any, Any], ne: tuple[int, ...]) -> np.ndarray[Any, Any]:
        ne0 = ne[0]
        nrows = prod(ne[1:])
        matrix = np.asarray(data).reshape(nrows, ne0)
        matrix_f32 = np.ascontiguousarray(matrix, dtype=np.float32)
        out = np.empty((nrows * self.row_size(ne0),), dtype=np.uint8)
        written = self.lib.higgs_ggml_quantize_chunk(
            GGML_TYPE_Q4_K,
            matrix_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.c_void_p),
            nrows,
            ne0,
        )
        if written != out.nbytes:
            raise RuntimeError(f"GGML wrote {written} bytes, expected {out.nbytes}")
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT, help="Input mixed/F16 GGUF.")
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT, help="Output mostly-Q4_K GGUF.")
    parser.add_argument("--min-elements", type=int, default=16384, help="Minimum tensor elements required for Q4_K conversion.")
    parser.add_argument(
        "--only-tensor",
        action="append",
        default=[],
        help="Debug/testing escape hatch: only quantize exact tensor name(s). May be passed more than once.",
    )
    parser.add_argument(
        "--quantize-token-embeddings",
        action="store_true",
        help="Also quantize token_embd/output embeddings. Disabled by default for runtime compatibility.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without writing output.")
    return parser.parse_args()


def field_value(field: Any) -> tuple[Any, GGUFValueType, GGUFValueType | None]:
    value = field.contents()
    vtype = field.types[0]
    subtype = field.types[-1] if vtype == GGUFValueType.ARRAY and len(field.types) > 1 else None
    return value, vtype, subtype


def should_quantize(
    name: str,
    ne: tuple[int, ...],
    tensor_type: GGMLQuantizationType,
    min_elements: int,
    quantize_token_embeddings: bool,
    only_tensors: set[str],
) -> bool:
    if only_tensors and name not in only_tensors:
        return False
    if tensor_type not in (GGMLQuantizationType.F16, GGMLQuantizationType.F32):
        return False
    if len(ne) != 2:
        return False
    if prod(ne) < min_elements:
        return False
    if ne[0] % GGML_QUANT_SIZES[GGMLQuantizationType.Q4_K][0] != 0:
        return False

    lowered = name.lower()
    if not quantize_token_embeddings and ("token_embd" in lowered or lowered == "output.weight"):
        return False
    return not any(part in lowered for part in SKIP_NAME_PARTS)


def build_plan(
    reader: GGUFReader,
    quantizer: GgmlQuantizer,
    min_elements: int,
    quantize_token_embeddings: bool,
    only_tensors: set[str],
) -> list[TensorPlan]:
    plan: list[TensorPlan] = []
    for tensor in reader.tensors:
        ne = tuple(int(x) for x in tensor.shape)
        shape = tuple(int(x) for x in tensor.data.shape)
        quantize = should_quantize(tensor.name, ne, tensor.tensor_type, min_elements, quantize_token_embeddings, only_tensors)
        output_type = GGMLQuantizationType.Q4_K if quantize else tensor.tensor_type
        output_nbytes = prod(ne[1:]) * quantizer.row_size(ne[0]) if quantize else int(tensor.n_bytes)
        plan.append(
            TensorPlan(
                name=tensor.name,
                shape=shape,
                ne=ne,
                input_type=tensor.tensor_type,
                output_type=output_type,
                input_nbytes=int(tensor.n_bytes),
                output_nbytes=int(output_nbytes),
                quantize=quantize,
            )
        )
    return plan


def copy_metadata(reader: GGUFReader, writer: GGUFWriter) -> None:
    writer.kv_data[0].clear()
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        value, vtype, subtype = field_value(field)
        writer.kv_data[0][key] = GGUFValue(value=value, type=vtype, sub_type=subtype)

    writer.kv_data[0]["general.file_type"] = GGUFValue(int(LlamaFileType.MOSTLY_Q4_K_M), GGUFValueType.UINT32, None)
    writer.kv_data[0]["general.quantized_by"] = GGUFValue("higgs-audio.cpp/scripts/quantize_higgs_gguf_q4_k.py", GGUFValueType.STRING, None)


def write_output(reader: GGUFReader, plan: list[TensorPlan], output: Path, arch: str, quantizer: GgmlQuantizer) -> None:
    writer = GGUFWriter(output, arch=arch, use_temp_file=False)
    writer.data_alignment = int(reader.alignment)
    copy_metadata(reader, writer)

    for item in plan:
        writer.tensors[-1][item.name] = TensorInfo(item.shape, item.output_type, item.output_nbytes, None)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    by_name = {item.name: item for item in plan}
    for index, tensor in enumerate(reader.tensors, start=1):
        item = by_name[tensor.name]
        out = quantizer.q4_k(tensor.data, item.ne) if item.quantize else np.asarray(tensor.data)
        writer.write_tensor_data(out)
        action = "q4_k" if item.quantize else "copy"
        print(
            f"[{index:04d}/{len(plan):04d}] {action:5s} {item.name} "
            f"{item.input_nbytes / 1024 / 1024:.1f}MiB -> {item.output_nbytes / 1024 / 1024:.1f}MiB",
            flush=True,
        )

    writer.close()


def main() -> None:
    args = parse_args()
    quantizer = GgmlQuantizer(PROJECT_ROOT)
    reader = GGUFReader(args.input)
    arch = reader.fields["general.architecture"].contents()
    if isinstance(arch, bytes):
        arch = arch.decode("utf-8")

    plan = build_plan(reader, quantizer, args.min_elements, args.quantize_token_embeddings, set(args.only_tensor))
    q_tensors = [item for item in plan if item.quantize]
    in_bytes = sum(item.input_nbytes for item in plan)
    out_bytes = sum(item.output_nbytes for item in plan)
    q_in_bytes = sum(item.input_nbytes for item in q_tensors)
    q_out_bytes = sum(item.output_nbytes for item in q_tensors)

    print(f"input:  {args.input}")
    print(f"output: {args.output}")
    print(f"tensors: {len(plan)} total, {len(q_tensors)} q4_k")
    print(f"tensor payload: {in_bytes / 1024**3:.2f}GiB -> {out_bytes / 1024**3:.2f}GiB")
    print(f"quantized subset: {q_in_bytes / 1024**3:.2f}GiB -> {q_out_bytes / 1024**3:.2f}GiB")

    if args.dry_run:
        for item in q_tensors[:80]:
            print(f"q4_k {item.name} ne={item.ne} {item.input_nbytes / 1024 / 1024:.1f}MiB -> {item.output_nbytes / 1024 / 1024:.1f}MiB")
        if len(q_tensors) > 80:
            print(f"... {len(q_tensors) - 80} more q4_k tensors")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = args.output.with_suffix(args.output.suffix + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()
    write_output(reader, plan, tmp_output, arch, quantizer)
    tmp_output.replace(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
