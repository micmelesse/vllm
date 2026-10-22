#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""THE BUILDS, FROM THE TUNED TABLE: what the compile instantiates is the build half
of every config the table lists on the archs built (its kernel, block, waves, tile and
quantization; the grid is the launch's, never compiled), each kernel's builds in the
order the file first names them. CMake runs this at configure (the table and this file
are configure dependencies), and the host file includes what it writes: one
`constexpr BuildConfig` array a template, `kBuilds_<kernel>`. A config of the table is
then always a build, and a build is always in the table. The pattern is vLLM's own:
Machete's `generate.py` and Marlin's `generate_kernels.py` write their kernel instances
at configure the same way (CMakeLists.txt).

    python3 builds.py <hip_custom_communicator.json> <out.inc> <arch>[;<arch>...]
        <host.cuh> [<candidates.json>]

A TUNE'S CANDIDATES (gp tune, exploring its space: never a serving build): a JSON
list of configs of the table's shape (a kernel and its build's fields; the grid is
ignored), compiled beside the table's builds, each a build of its own file. A file
NAMES ITS BUILD (`hip_custom_communicator_<kernel>__<block>_w<waves>_t<tile>_<q>.hip`;
the manifest `hip_custom_communicator_instances.json` says each file's builds: every
build whose instances it holds), so a rebuild with other candidates recompiles only the
new ones. A candidate the compiler refuses (a kernel's static_assert) is the tuner's to
drop from its candidates: dev.sh's probe compiles every file and keeps the candidates
whose files compiled.
"""

import json
import re
import sys
from pathlib import Path

# A BUILD'S FIELDS, in the table's order.
FIELDS = (
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


def family(kernel: str) -> str:
    """The op's BuildConfig a kernel's builds are (a fused op's, or its norm's)."""
    if kernel.endswith(("add_attn_res_rms_norm", "add_attn_res_rms_norm_rows")):
        return "AllReduceAddAttnResRmsNormBuildConfig"
    if kernel.endswith(("add_rms_norm", "add_rms_norm_rows")):
        return "AllReduceAddRmsNormBuildConfig"
    if kernel.endswith(("rms_norm", "rms_norm_rows")):
        return "AllReduceRmsNormBuildConfig"
    return "AllReduceBuildConfig"


def builds(table: dict, archs: list[str]) -> dict[str, list[tuple]]:
    """Each kernel's builds, deduplicated, in the order the file first names them."""
    out: dict[str, list[tuple]] = {}
    for op in table.values():
        for arch in archs:
            for c in op["configs"].get(arch, []):
                b = tuple(c[f] for f in FIELDS)
                if b not in out.setdefault(c["kernel"], []):
                    out[c["kernel"]].append(b)
    return out


def literal(kernel: str, b: tuple) -> str:
    bx, by, bz, waves, tm, tn, tk, q = b
    fields = [f"{{{bx}, {by}, {bz}}}", str(waves), str(tm), str(tn)]
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
PLAIN = (
    "all_reduce_pull_one_shot",
    "all_reduce_pull_two_shot",
    "all_reduce_push_one_shot",
    "all_reduce_push_two_shot",
)
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
        r"C\.kernel == Kernel::(\w+)\)\s*return reinterpret_cast<const void\*>\(\s*"
        r"(\w+)<(.*?)>\);",
        body,
        re.S,
    ):
        assert m.group(1) == m.group(2), m.group(0)
        out[m.group(1)] = " ".join(m.group(3).split())
    return out


def lays_out(kernel: str, b: tuple, world: int) -> bool:
    """Whether a plain all-reduce build lays out whole (the host's one_shot_lays_out and
    two_shot_lays_out, at a 16-bit dtype); every other kernel's builds do."""
    block, _, _, _, tile_m, tile_n, _, _ = b
    if kernel == "all_reduce_push_one_shot":
        nl = LINE_PAYLOAD_BYTES // 2
    elif kernel in PLAIN:
        nl = PACK_BYTES // 2
    else:
        return True
    if "one_shot" in kernel:
        return block % tile_m == 0 and tile_n % (nl * (block // tile_m)) == 0
    return (
        tile_m == 1
        and block % world == 0
        and (block // world) % WAVE_SIZE == 0
        and tile_n % (nl * (block // world)) == 0
    )


def instances(kernel: str, args: str, b: tuple) -> list[str]:
    """EVERY INSTANCE OF A BUILD the dispatch reaches (the host's `with_config_build`):
    each dtype and world, the weight's dtype where a norm takes one, contiguous or not
    (the plain two-shots), chunked or not (every kernel but the plain all-reduce's),
    with a tail or not (an INT4 two-shot); one that does not lay out has none."""
    block, _, _, waves, tile_m, tile_n, tile_k, q = b
    two_shot = "two_shot" in kernel
    out: list[str] = []
    for _, t in DTYPES:
        for world in WORLDS:
            if not lays_out(kernel, b, world):
                continue
            weighted = (
                kernel.endswith(("rms_norm", "rms_norm_rows"))
                and "attn_res" not in kernel
            )
            for w in (t, "float") if weighted else (t,):
                for contiguous in (
                    ("true", "false") if kernel in PLAIN and two_shot else ("false",)
                ):
                    for chunked in ("false",) if kernel in PLAIN else ("true", "false"):
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
                                "C.contiguous": contiguous,
                                "C.chunked": chunked,
                                "C.tail": tail,
                            }
                            got = ", ".join(values[a.strip()] for a in args.split(","))
                            if f"{kernel}<{got}>" not in out:
                                out.append(f"{kernel}<{got}>")
    return out


def main() -> None:
    table_path, out_path, arch_list = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    host = Path(sys.argv[4]).read_text() if len(sys.argv) > 4 else None
    archs = [a for a in arch_list.replace(",", ";").split(";") if a]
    table = json.loads(Path(table_path).read_text())
    if len(sys.argv) > 5 and sys.argv[5]:
        # THE CANDIDATES, after the table's own: an op's table holds a kernel's configs
        # on each arch.
        extra = {
            "candidates": {
                "configs": {a: json.loads(Path(sys.argv[5]).read_text()) for a in archs}
            }
        }
        table = {**table, **extra}
    every = builds(table, archs)
    text = [
        "// GENERATED by csrc/rocm/hip_custom_communicator/builds.py from the tuned",
        f"// table (hip_custom_communicator.json) on {', '.join(archs)}: every",
        "// config's build. Do not edit; edit the table's configs (a config no input",
        "// names is a candidate the tuner searches).",
        "",
    ]
    for kernel, bs in every.items():
        text.append(f"constexpr BuildConfig kBuilds_{kernel}[] = {{")
        text += [literal(kernel, b) for b in bs]
        text += ["};", ""]
    write(out_path, "\n".join(text))
    if host is None:
        return
    # THE INSTANCES, ONE FILE A BUILD (compiled in parallel, cached each), and the
    # header the host includes so that it instantiates none itself: each declared
    # `extern template` by its own type.
    args = arguments(host)
    out_dir = out_path.parent
    declared = [
        "// GENERATED by builds.py: every kernel instance, compiled in its build's own",
        "// file (hip_custom_communicator_<kernel>__<build>.hip); declared here so the host",
        "// instantiates none.",
        f"static_assert(kTarget.wave_size == {WAVE_SIZE} && "
        f"kBuild.memory.pack_bytes == "
        f"{PACK_BYTES}, \"builds.py's wave and pack are not the host's\");",
        "static_assert(kBuild.supports.worlds == std::array<int, 3>{"
        + ", ".join(map(str, WORLDS))
        + "}, \"builds.py's worlds are not the host's\");",
        "static_assert(kBuild.supports.dtypes == std::array<DType, 2>{"
        + ", ".join(f"DType::{d}" for d, _ in DTYPES)
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
    bx, by, bz, waves, tm, tn, tk, q = b
    return (
        f"hip_custom_communicator_{kernel}__{bx}x{by}x{bz}_w{waves}"
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
