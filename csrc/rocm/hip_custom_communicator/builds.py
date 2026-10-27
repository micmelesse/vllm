#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""THE BUILDS, FROM THE TUNED TABLES: what the compile instantiates is the build half
of every config the tables of the archs built list (`hip_custom_communicator.json`, a
table an arch, each gd's `TunedTable`: its `configs`, each its op, its `kernel` the op's
one kernel, its `algorithm` which of the kernel's bodies: `one_shot_pull`,
`one_shot_push`, `two_shot_pull`, `two_shot_push` or `rows`, its block, waves, tile,
quantization and pack; the grid is the launch's, never compiled), each kernel's builds in the
order the tables first name them. THE BUILD MAKES WHAT IT IS TOLD (Michael, October 25):
nothing is left out and nothing added; every kernel takes any config it is built at
(its tails its own), and what an algorithm cannot do at a call (INT4 on a one-shot, a
pushed one-shot's tile past its slot) is refused by name at the call
(`hip_custom_communicator.cu`'s checks), never by the build. CMake runs this at
configure (the table and this file
are configure dependencies), and the host file includes what it writes: one
`constexpr BuildConfig` array a template, `kBuilds_<kernel>`. A config of the table is
then always a build, and a build is always in the table. The pattern is vLLM's own:
Machete's `generate.py` and Marlin's `generate_kernels.py` write their kernel instances
at configure the same way (CMakeLists.txt).

    python3 builds.py <hip_custom_communicator.json> <out.inc> <arch>[;<arch>...] <hip_custom_communicator.cu>

A TUNE'S CANDIDATES ARE THE TABLE'S TOO: configs no input picks, which a tune adds on
its own copy of the table; gd's tune prunes them when it writes. A file NAMES ITS BUILD
(`hip_custom_communicator_<kernel>_<algorithm>__<block>_w<waves>_t<tile>_<q>_p<pack>.hip`;
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
    "pack_bytes",
)
QUANTIZATIONS = {"exact": "Quantization::exact", "int4": "Quantization::int4"}


# EACH KERNEL'S BuildConfig, its op's (one type an op), and its ALGORITHM's enum, its op file's;
# a norm on rows has one algorithm (the table's "rows") and no enum.
BUILDS = {
    "all_reduce": "AllReduceBuildConfig",
    "all_reduce_rms_norm": "AllReduceRmsNormBuildConfig",
    "all_reduce_add_rms_norm": "AllReduceAddRmsNormBuildConfig",
    "all_reduce_add_attn_res_rms_norm": "AllReduceAddAttnResRmsNormBuildConfig",
    "rms_norm_rows": "RmsNormBuildConfig",
    "add_rms_norm_rows": "AddRmsNormBuildConfig",
    "add_attn_res_rms_norm_rows": "AddAttnResRmsNormBuildConfig",
}
ALGORITHM_ENUMS = {
    "all_reduce": "AllReduceAlgorithm",
    "all_reduce_rms_norm": "AllReduceRmsNormAlgorithm",
    "all_reduce_add_rms_norm": "AllReduceAddRmsNormAlgorithm",
    "all_reduce_add_attn_res_rms_norm": "AllReduceAddAttnResRmsNormAlgorithm",
}
# THE KERNELS WITH A K (TILE_K, AttnRes's sources a step).
WITH_K = ("all_reduce_add_attn_res_rms_norm", "add_attn_res_rms_norm_rows")


def tables(path: Path, archs: list[str]) -> list[dict]:
    """The tuned table of each arch built that has one, in the archs' order, as it is."""
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


def algorithm_value(kernel: str, algorithm: str) -> str:
    """A build's algorithm as its kernel's template argument (its op file's enum)."""
    return f"{ALGORITHM_ENUMS[kernel]}::{algorithm}"


def literal(kernel: str, b: tuple) -> str:
    algorithm, bx, by, bz, waves, tm, tn, tk, q, pack = b
    fields = [algorithm_value(kernel, algorithm)] if kernel in ALGORITHM_ENUMS else []
    fields += [f"{{{bx}, {by}, {bz}}}", str(waves), str(tm), str(tn)]
    if kernel in WITH_K:
        fields.append(str(tk))
    elif tk != 1:
        raise SystemExit(f"{kernel}: tile_k {tk}, but the kernel has no K")
    fields += [QUANTIZATIONS[q], str(pack)]
    return f"    {BUILDS[kernel]}{{{', '.join(fields)}}},"


# THE MACHINE'S FACTS AND WHAT IS COMPILED, as the host holds them (kWaveSize, kPackBytes,
# kWorlds, kDtypes: PACK_BYTES the widest pack, a build's own its config's); the generated
# header asserts them equal, so a change there fails the compile naming this file.
WAVE_SIZE, PACK_BYTES = 64, 16
DTYPES = (("f16", "c10::Half"), ("bf16", "c10::BFloat16"))
SCALAR_TYPES = {"f16": "Half", "bf16": "BFloat16"}
WORLDS = (2, 4, 8)
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
        r"B\.op == OpType::\w+\)\s*return reinterpret_cast<const void\*>\(\s*"
        r"(\w+)<(.*?)>\);",
        body,
        re.S,
    ):
        out[m.group(1)] = " ".join(m.group(2).split())
    return out


def algorithms(host: str) -> dict[str, list[str]]:
    """Each kernel's algorithms by name as its op's build lists them (its `algorithms`: the
    bodies its op file has; a norm on rows has one, "rows"), by kernel."""
    out = {}
    for kernel, build in BUILDS.items():
        body = host[host.index(f"struct {build} {{") :]
        body = body[: body.index("\n};\n")]
        named = re.search(r"algorithms\[\] = \{(.*?)\};", body, re.S)
        out[kernel] = re.findall(r'"(\w+)"', named.group(1)) if named else ["rows"]
    return out


def instances(kernel: str, args: str, b: tuple) -> list[str]:
    """EVERY INSTANCE OF A BUILD the dispatch reaches (the host's `with_config_build`):
    each dtype and world, the weight's dtype where a norm takes one, contiguous or not
    (the plain two-shots), chunked or not (every kernel but the plain all-reduce's),
    with a tail or not (an INT4 two-shot, or any INT4 build of the plain all-reduce), and the
    plain all-reduce's INT4 build quantizing or passing the call exact."""
    algorithm, block, _, _, waves, tile_m, tile_n, tile_k, q, pack = b
    # AN INT4 BUILD AT A NARROWER PACK compiles its exact body (the host's
    # compiled_quantization: INT4's word is a pack of 8 values; the call is refused).
    if pack != PACK_BYTES:
        q = "exact"
    two_shot = algorithm in ("two_shot_pull", "two_shot_push")
    plain = kernel == "all_reduce"
    out: list[str] = []
    for _, t in DTYPES:
        for world in WORLDS:
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
                        # (tail, quantize_int4): an INT4 two-shot with a tail or not; the plain
                        # all-reduce's INT4 build also launched exact (its exact instance, the
                        # exact build's at that shape); a fused op's INT4 build always moves INT4.
                        for tail, quantize_int4 in (
                            (("true", "true"), ("false", "true"), ("false", "false"))
                            if q == "int4" and plain
                            else (("true", "true"), ("false", "true"))
                            if q == "int4" and two_shot
                            else (("false", "true" if q == "int4" else "false"),)
                        ):
                            values = {
                                "T": t,
                                "W": w,
                                "I.world": str(world),
                                "B.tile_m": str(tile_m),
                                "B.tile_n": str(tile_n),
                                "B.tile_k": str(tile_k),
                                "B.block.x": str(block),
                                "I.wave_size": str(WAVE_SIZE),
                                "B.pack_bytes": str(pack),
                                "B.waves_per_eu": str(waves),
                                "I.contiguous": contiguous,
                                "I.chunked": chunked,
                                "I.tail": tail,
                                "I.quantize_int4": quantize_int4,
                            }
                            if kernel in ALGORITHM_ENUMS:
                                values["B.algorithm"] = algorithm_value(kernel, algorithm)
                            got = ", ".join(values[a.strip()] for a in args.split(","))
                            if f"{kernel}<{got}>" not in out:
                                out.append(f"{kernel}<{got}>")
    return out


def main() -> None:
    table_path, out_path, arch_list = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    host = Path(sys.argv[4]).read_text()
    archs = [a for a in arch_list.replace(",", ";").split(";") if a]
    found = tables(table_path, archs)
    # EVERY BUILD THE TABLE NAMES AT ONE OF ITS KERNEL'S ALGORITHMS (its op's build's
    # `algorithms`: a body the kernel has, or the config names nothing to compile).
    has = algorithms(host)
    for kernel, bs in named(found).items():
        for b in bs:
            if kernel not in has or b[0] not in has[kernel]:
                raise SystemExit(f"{kernel} has no {b[0]} algorithm: {b}")
    every = named(found)
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
        f"static_assert(kWaveSize == {WAVE_SIZE} && "
        f"kPackBytes == "
        f"{PACK_BYTES}, \"builds.py's wave and pack are not the host's\");",
        "static_assert(kWorlds == std::array<int, 3>{"
        + ", ".join(map(str, WORLDS))
        + "}, \"builds.py's worlds are not the host's\");",
        "static_assert(kDtypes == std::array<at::ScalarType, 2>{"
        + ", ".join(f"at::ScalarType::{SCALAR_TYPES[d]}" for d, _ in DTYPES)
        + "}, \"builds.py's dtypes are not the host's\");",
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
    algorithm, bx, by, bz, waves, tm, tn, tk, q, pack = b
    return (
        f"hip_custom_communicator_{kernel}_{algorithm}__{bx}x{by}x{bz}_w{waves}"
        f"_t{tm}x{tn}x{tk}_{q}_p{pack}.hip"
    )


def write(path: Path, text: str) -> None:
    """`text` into `path`, untouched where it already holds it (so a reconfigure
    recompiles only what changed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


if __name__ == "__main__":
    main()
