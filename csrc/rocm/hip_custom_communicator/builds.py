#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""THE BUILDS, FROM THE TUNED TABLES: what the compile instantiates is the build half
of every config the tables of the archs built list (`hip_custom_communicator.json`, a
table an arch, each gd's `TunedTable`: its `configs`, each its op, its `kernel` the op's
one kernel, its `algorithm` which of the kernel's bodies: `one_shot_pull`,
`one_shot_push`, `two_shot_pull`, `two_shot_push` or `rows`, its block, waves, tile and
quantization; the grid is the launch's, never compiled), each kernel's builds in the
order the tables first name them. EVERY ALGORITHM OF A SHAPE (Michael,
October 23): for every build shape the table holds of a kernel (its block, waves, tile
and quantization: all a build is but its algorithm), the kernel is built at every
algorithm it has (the host's `kAlgorithms_<kernel>`) that takes that shape (`takes`: a
refusal the kernel's math or the hardware forces, which the communicator refuses by
name at a call), the table's own builds first and in its order. So a tune's prune,
which drops every config no input names, can drop a shape but never an algorithm of
one: every path a call can be forced to take stays compiled. CMake runs this at
configure (the table and this file
are configure dependencies), and the host file includes what it writes: one
`constexpr BuildConfig` array a template, `kBuilds_<kernel>`. A config of the table is
then always a build, and a build is always in the table. The pattern is vLLM's own:
Machete's `generate.py` and Marlin's `generate_kernels.py` write their kernel instances
at configure the same way (CMakeLists.txt).

    python3 builds.py <hip_custom_communicator.json> <out.inc> <arch>[;<arch>...] <host.cuh>

A TUNE'S CANDIDATES ARE THE TABLE'S TOO: configs no input picks, which a tune adds on
its own copy of the table; gd's tune prunes them when it writes. A file NAMES ITS BUILD
(`hip_custom_communicator_<kernel>_<algorithm>__<block>_w<waves>_t<tile>_<q>.hip`;
the manifest
`hip_custom_communicator_instances.json` says each file's builds: every build whose
instances it holds), so a rebuild with other candidates recompiles only the new ones,
and a candidate the compiler refuses (a kernel's static_assert) is found by its file.
"""

import json
import re
import sys
from pathlib import Path

# A BUILD'S FIELDS, in the table's order.
FIELDS = (
    "algorithm",
    "block_x",
    "block_y",
    "block_z",
    "waves_per_eu",
    "tile_m",
    "tile_n",
    "tile_k",
    "quantization",
)
QUANTIZATIONS = {"exact": "Quantization::exact", "int4": "Quantization::int4"}
# THE ALGORITHMS, as common.cuh names them and numbers them (its template ints).
ALGORITHM_VALUES = {"one_shot_pull": 0, "one_shot_push": 1, "two_shot_pull": 2,
                    "two_shot_push": 3, "rows": 4}


# EACH KERNEL'S BuildConfig (a fused op's, or its norm's: one type an op family).
FAMILIES = {
    "all_reduce": "AllReduceBuildConfig",
    "all_reduce_rms_norm": "AllReduceRmsNormBuildConfig",
    "rms_norm_rows": "AllReduceRmsNormBuildConfig",
    "all_reduce_add_rms_norm": "AllReduceAddRmsNormBuildConfig",
    "add_rms_norm_rows": "AllReduceAddRmsNormBuildConfig",
    "all_reduce_add_attn_res_rms_norm": "AllReduceAddAttnResRmsNormBuildConfig",
    "add_attn_res_rms_norm_rows": "AllReduceAddAttnResRmsNormBuildConfig",
}


def family(kernel: str) -> str:
    """The op's BuildConfig a kernel's builds are (a fused op's, or its norm's)."""
    return FAMILIES[kernel]


def tables(path: Path, archs: list[str]) -> list[dict]:
    """The tuned table of each arch built that has one, in the archs' order."""
    file = json.loads(path.read_text())
    return [file[arch] for arch in archs if arch in file]


def named(tables: list[dict]) -> dict[str, list[tuple]]:
    """Each kernel's builds the tables' configs name, deduplicated, in the order the tables
    first name them."""
    out: dict[str, list[tuple]] = {}
    for table in tables:
        for c in table["configs"]:
            b = tuple(c[f] for f in FIELDS)
            if b not in out.setdefault(c["kernel"], []):
                out[c["kernel"]].append(b)
    return out


def takes(kernel: str, b: tuple) -> bool:
    """WHETHER `kernel` AT ITS ALGORITHM TAKES THE SHAPE: no where the kernel refuses it by
    construction, as the communicator refuses a call of it by name. INT4 is a two-shot's
    and not the plain pull's (`quantization_not_supported`: a one-shot reads its peers'
    inputs as they are); the pushed one-shot's tile is a row that fits its unfenced
    slot (`push_one_shot_tiles_fit`); a pulled fused two-shot's INT4 tile is whole groups
    a thread, a row high (its kernel's static_asserts: holes (a) and (b) of the task
    list); a plain build lays out at some built world (`lays_out`)."""
    algorithm, block, _, _, _, tile_m, tile_n, _, q = b
    if q == "int4" and not (
        algorithm == "two_shot_push" or (algorithm == "two_shot_pull" and kernel != "all_reduce")
    ):
        return False
    if kernel == "all_reduce" and algorithm == "one_shot_push":
        if tile_m != 1 or tile_n * 2 > UNFENCED_SLOT_BYTES:
            return False
    if (
        q == "int4"
        and kernel in FUSED
        and algorithm == "two_shot_pull"
        and (tile_n % (INT4_GROUP * block) != 0 or tile_m != 1)
    ):
        return False
    return any(lays_out(kernel, b, world) for world in WORLDS)


def builds(tables: list[dict], has: dict[str, list[str]]) -> dict[str, list[tuple]]:
    """Each kernel's builds: the tables', in their order, then every algorithm `has` lists
    of every shape the tables hold of the kernel, where the algorithm takes it (`takes`)."""
    out = named(tables)
    for kernel, bs in out.items():
        shapes = list(dict.fromkeys(b[1:] for b in bs))
        for shape in shapes:
            for algorithm in has[kernel]:
                b = (algorithm, *shape)
                if b not in bs and takes(kernel, b):
                    bs.append(b)
    return out


def literal(kernel: str, b: tuple) -> str:
    algorithm, bx, by, bz, waves, tm, tn, tk, q = b
    fields = [f"Algorithm::{algorithm}"]
    fields += [f"{{{bx}, {by}, {bz}}}", str(waves), str(tm), str(tn)]
    if family(kernel) == "AllReduceAddAttnResRmsNormBuildConfig":
        fields.append(str(tk))
    elif tk != 1:
        raise SystemExit(f"{kernel}: tile_k {tk}, but the kernel has no K")
    fields.append(QUANTIZATIONS[q])
    return f"    {family(kernel)}{{{', '.join(fields)}}},"


# THE MACHINE'S FACTS AND WHAT IS COMPILED, as the host holds them (kTarget, kBuild);
# the generated header asserts them equal, so a change there fails the compile naming
# this file.
WAVE_SIZE, PACK_BYTES = 64, 16
DTYPES = (("f16", "c10::Half"), ("bf16", "c10::BFloat16"))
WORLDS = (2, 4, 8)
LINE_PAYLOAD_BYTES = (
    8  # the pushed one-shot's pack: a line's payload (common.cuh's kLinePayloadBytes)
)
UNFENCED_SLOT_BYTES = 2048  # a pushed one-shot block's tile, at most (kUnfencedSlotBytes)
INT4_GROUP = 32  # an INT4 group's elements (common.cuh's kInt4Group)
FUSED = ("all_reduce_rms_norm", "all_reduce_add_rms_norm", "all_reduce_add_attn_res_rms_norm")
HEADERS = (
    "kernels/all_reduce.cuh",
    "kernels/all_reduce_add_attn_res_rms_norm.cuh",
    "kernels/all_reduce_add_rms_norm.cuh",
)


def arguments(host: str) -> dict[str, str]:
    """Each kernel's template arguments as `kernel_pointer` names them (the host's one
    list of every kernel's template arguments), by kernel."""
    body = host[host.index("const void* kernel_pointer() {") :]
    body = body[: body.index("\n}\n")]
    out = {}
    for m in re.finditer(
        r"C\.kernel == Template::(\w+)\)\s*return reinterpret_cast<const void\*>\(\s*"
        r"(\w+)<(.*?)>\);",
        body,
        re.S,
    ):
        assert m.group(1) == m.group(2), m.group(0)
        out[m.group(1)] = " ".join(m.group(3).split())
    return out


def algorithms(host: str) -> dict[str, list[str]]:
    """Each kernel's algorithms as the host lists them (`kAlgorithms_<kernel>`: the bodies its
    op file has), by kernel."""
    out = {}
    for m in re.finditer(r"constexpr int kAlgorithms_(\w+)\[\] = \{(.*?)\};", host, re.S):
        out[m.group(1)] = re.findall(r"Algorithm::(\w+)", m.group(2))
    return out


def lays_out(kernel: str, b: tuple, world: int) -> bool:
    """Whether a plain all-reduce build lays out (the host's one_shot_lays_out and
    two_shot_lays_out, at a 16-bit dtype, whose comment says why each rule is forced):
    any block walks any tile of whole packs (a step's packs past the tile masked, a
    rank's lanes part of a wave where the block is narrow); a two-shot's tile has no
    rows (the flat walk) and its block is WORLD rows of threads. Every other kernel's
    builds do, but for an INT4 pulled norm or AttnRes whose tile is not whole groups a
    thread (TILE_N a multiple of 32 * THREADS), which its kernel's static_assert refuses
    by name (a thread quantizes whole groups alone)."""
    algorithm, block, _, _, _, tile_m, tile_n, _, _ = b
    if kernel != "all_reduce":
        return True
    if algorithm == "one_shot_push":
        nl = LINE_PAYLOAD_BYTES // 2
    else:
        nl = PACK_BYTES // 2
    if algorithm in ("one_shot_pull", "one_shot_push"):
        return tile_n % nl == 0
    return tile_m == 1 and block % world == 0 and tile_n % nl == 0


def instances(kernel: str, args: str, b: tuple) -> list[str]:
    """EVERY INSTANCE OF A BUILD the dispatch reaches (the host's `with_config_build`):
    each dtype and world, the weight's dtype where a norm takes one, contiguous or not
    (the plain two-shots), chunked or not (every kernel but the plain all-reduce's),
    with a tail or not (an INT4 two-shot); one that does not lay out has none."""
    algorithm, block, _, _, waves, tile_m, tile_n, tile_k, q = b
    two_shot = algorithm in ("two_shot_pull", "two_shot_push")
    plain = kernel == "all_reduce"
    out: list[str] = []
    for _, t in DTYPES:
        for world in WORLDS:
            if not lays_out(kernel, b, world):
                continue
            weighted = kernel in (
                "all_reduce_rms_norm",
                "rms_norm_rows",
                "all_reduce_add_rms_norm",
                "add_rms_norm_rows",
            )
            for w in (t, "float") if weighted else (t,):
                for contiguous in (
                    ("true", "false") if plain and two_shot else ("false",)
                ):
                    for chunked in ("false",) if plain else ("true", "false"):
                        for tail in (
                            ("true", "false")
                            if q == "int4" and two_shot
                            else ("false",)
                        ):
                            values = {
                                "T": t,
                                "W": w,
                                "C.world": str(world),
                                "C.tile_m": str(tile_m),
                                "C.tile_n": str(tile_n),
                                "C.tile_k": str(tile_k),
                                "C.block.x": str(block),
                                "C.wave_size": str(WAVE_SIZE),
                                "C.pack_bytes": str(PACK_BYTES),
                                "C.waves_per_eu": str(waves),
                                "C.quantization": "1" if q == "int4" else "0",
                                "C.algorithm": str(ALGORITHM_VALUES[algorithm]),
                                "C.contiguous": contiguous,
                                "C.chunked": chunked,
                                "C.tail": tail,
                            }
                            got = ", ".join(values[a.strip()] for a in args.split(","))
                            if f"{kernel}<{got}>" not in out:
                                out.append(f"{kernel}<{got}>")
    return out


def main() -> None:
    table_path, out_path, arch_list = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    host = Path(sys.argv[4]).read_text()
    archs = [a for a in arch_list.replace(",", ";").split(";") if a]
    found = tables(table_path, archs)
    # EVERY BUILD THE TABLE NAMES AT ONE OF ITS KERNEL'S ALGORITHMS (the host's
    # kAlgorithms_<kernel>; its own static_assert says the same at compile time, this says
    # it naming the config).
    has = algorithms(host)
    for kernel, bs in named(found).items():
        for b in bs:
            if kernel not in has or b[0] not in has[kernel]:
                raise SystemExit(f"{kernel} has no {b[0]} algorithm: {b}")
    every = builds(found, has)
    text = [
        "// GENERATED by csrc/rocm/hip_custom_communicator/builds.py from the tuned",
        f"// tables (hip_custom_communicator.json) on {', '.join(archs)}: every",
        "// config's build. Do not edit; edit the table's configs (a config no input",
        "// names is a candidate the tuner searches).",
        "",
    ]
    for kernel, bs in every.items():
        text.append(f"constexpr BuildConfig kBuilds_{kernel}[] = {{")
        text += [literal(kernel, b) for b in bs]
        text += ["};", ""]
    write(out_path, "\n".join(text))
    # THE INSTANCES, ONE FILE A BUILD (compiled in parallel, cached each), and the
    # header the host includes so that it instantiates none itself: each declared
    # `extern template` by its own type.
    args = arguments(host)
    out_dir = out_path.parent
    declared = [
        "// GENERATED by builds.py: every kernel instance, compiled in its build's own",
        "// file (hip_custom_communicator_<kernel>_<algorithm>__<build>.hip);",
        "// declared here so the host instantiates none.",
        f"static_assert(kTarget.wave_size == {WAVE_SIZE} && "
        f"kBuild.memory.pack_bytes == "
        f"{PACK_BYTES}, \"builds.py's wave and pack are not the host's\");",
        "static_assert(kBuild.supports.worlds == std::array<int, 3>{"
        + ", ".join(map(str, WORLDS))
        + "}, \"builds.py's worlds are not the host's\");",
        "static_assert(kBuild.supports.dtypes == std::array<DType, 2>{"
        + ", ".join(f"DType::{d}" for d, _ in DTYPES)
        + "}, \"builds.py's dtypes are not the host's\");",
        "static_assert("
        + " && ".join(f"Algorithm::{a} == {v}" for a, v in ALGORITHM_VALUES.items())
        + ", \"builds.py's algorithms are not common.cuh's\");",
        f"static_assert(kUnfencedSlotBytes == {UNFENCED_SLOT_BYTES} && kInt4Group == "
        f"{INT4_GROUP}, \"builds.py's unfenced slot and INT4 group are not the host's\");",
        "",
    ]
    files = []
    manifest: dict[str, list[dict]] = {}
    held: dict[str, str] = {}
    for kernel, bs in every.items():
        for b in bs:
            # AN INSTANCE ONCE: builds that differ only in a field the kernel takes no
            # parameter for (a push norm's tile_m) are the same instances, in the file
            # of the build that named them first; the manifest lists each build under
            # every file holding its instances.
            needed = instances(kernel, args[kernel], b)
            got = [i for i in needed if i not in held]
            name = file_of(kernel, b)
            held.update((i, name) for i in got)
            for f in dict.fromkeys(held[i] for i in needed):
                manifest.setdefault(f, []).append({"kernel": kernel, **dict(zip(FIELDS, b))})
            if not got:
                continue
            files.append(name)
            body = [
                "// GENERATED by builds.py: "
                + kernel
                + " at "
                + literal(kernel, b).strip(),
                "#include <c10/util/BFloat16.h>",
                "#include <c10/util/Half.h>",
                "#include <hip/hip_runtime.h>",
                "",
            ]
            body += [f'#include "hip_custom_communicator/{h}"' for h in HEADERS]
            body += ["", "namespace hip_custom_communicator {"]
            body += [f"template __global__ decltype({i}) {i};" for i in got]
            body += ["}  // namespace hip_custom_communicator", ""]
            write(out_dir / name, "\n".join(body))
            declared += [f"extern template __global__ decltype({i}) {i};" for i in got]
    write(out_dir / "hip_custom_communicator_instances.inc", "\n".join(declared) + "\n")
    write(out_dir / "hip_custom_communicator_instances.txt", "\n".join(files) + "\n")
    write(
        out_dir / "hip_custom_communicator_instances.json",
        json.dumps(manifest, indent=1) + "\n",
    )


def file_of(kernel: str, b: tuple) -> str:
    """The file a build's instances are compiled in, named by the build."""
    algorithm, bx, by, bz, waves, tm, tn, tk, q = b
    return (
        f"hip_custom_communicator_{kernel}_{algorithm}__{bx}x{by}x{bz}_w{waves}"
        f"_t{tm}x{tn}x{tk}_{q}.hip"
    )


def write(path: Path, text: str) -> None:
    """`text` into `path`, untouched where it already holds it (so a reconfigure
    recompiles only what changed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


if __name__ == "__main__":
    main()
