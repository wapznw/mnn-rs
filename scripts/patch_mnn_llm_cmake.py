#!/usr/bin/env python3
"""Idempotently patch MNN's LLM engine CMakeLists.txt.

Upstream MNN bug (present in 3.6.1 and current main):
`transformers/llm/engine/CMakeLists.txt` attaches a POST_BUILD custom command
to target `llm`, but when MNN_SEP_BUILD is OFF the target is created as an
OBJECT library. Modern CMake rejects this:

    Target "llm" is an OBJECT library that may not have PRE_BUILD, PRE_LINK,
    or POST_BUILD commands.

Workaround: when MNN_SEP_BUILD is OFF (OBJECT library), copy the headers at
configure time with `file(COPY ...)` instead of POST_BUILD. The POST_BUILD
path is preserved for SEP_BUILD builds (SHARED/STATIC target, where it is
legal).

Usage: patch_mnn_llm_cmake.py <path-to-mnn-source>

Safe to run multiple times (idempotent). Skips silently when the file or the
expected pattern is absent (e.g. MNN versions without the LLM engine, or a
future version that already fixed the issue).
"""

import pathlib
import sys

REL_PATH = pathlib.PurePath("transformers/llm/engine/CMakeLists.txt")

NEEDLE = """IF(NOT NATIVE_INCLUDE_OUTPUT)
  set(NATIVE_INCLUDE_OUTPUT ".")
ENDIF()
add_custom_command(
  TARGET llm
  POST_BUILD
  COMMAND ${CMAKE_COMMAND}
  ARGS -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/include ${NATIVE_INCLUDE_OUTPUT}
)
ELSE()"""

REPLACEMENT = """IF(NOT NATIVE_INCLUDE_OUTPUT)
  set(NATIVE_INCLUDE_OUTPUT ".")
ENDIF()
IF(NOT MNN_SEP_BUILD)
  # mnn-rs workaround for upstream MNN bug: `llm` is an OBJECT library when
  # MNN_SEP_BUILD is OFF, and CMake forbids POST_BUILD on OBJECT libraries.
  # Copy headers at configure time instead.
  file(COPY ${CMAKE_CURRENT_LIST_DIR}/include/ DESTINATION ${NATIVE_INCLUDE_OUTPUT})
ELSE()
add_custom_command(
  TARGET llm
  POST_BUILD
  COMMAND ${CMAKE_COMMAND}
  ARGS -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/include ${NATIVE_INCLUDE_OUTPUT}
)
ENDIF()
ELSE()"""

MARKER = "file(COPY ${CMAKE_CURRENT_LIST_DIR}/include/ DESTINATION"


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path-to-mnn-source>", file=sys.stderr)
        return 2

    cmake_file = pathlib.Path(sys.argv[1]) / REL_PATH
    if not cmake_file.exists():
        print(f"skip: {cmake_file} not found (no LLM engine in this MNN version)")
        return 0

    original = cmake_file.read_text(encoding="utf-8")

    if MARKER in original:
        print(f"skip: {cmake_file} already patched")
        return 0

    if NEEDLE not in original:
        print(f"warn: expected pattern not found in {cmake_file}, skipping")
        return 0

    cmake_file.write_text(original.replace(NEEDLE, REPLACEMENT), encoding="utf-8")
    print(f"patched: {cmake_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
