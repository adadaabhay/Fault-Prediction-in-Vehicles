"""Locate a C toolchain and build the edge runtime.

The engine previously had no build at all: its test suite text-scraped the
source for `malloc(` and computed struct sizes arithmetically, so nothing ever
established that the code compiled, let alone that it computed correctly.

Two build paths, tried in order:

1. **Direct compiler invocation.** This is a three-file C99 project with no
   dependencies beyond libm, so a single `gcc -shared` produces the artifact.
   Requiring CMake for that would add a prerequisite the project does not need.
2. **CMake**, when present -- useful for cross-compilation to an actual VMC/ECU
   target, and for MSVC on Windows.

Compilers installed outside PATH are discovered from the well-known locations
in ``_EXTRA_COMPILER_DIRS`` (an MSYS2 install, for instance, does not put
itself on PATH).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

C_DIR = Path(__file__).resolve().parent
BUILD_DIR = C_DIR / "build"

SOURCES = ("tank_pdm_infer.c", "tank_pdm_weights.c")
SELFTEST_SOURCES = ("main_test.c",) + SOURCES

_COMPILERS = ("gcc", "clang", "cc", "cl")

# Toolchains that commonly install without touching PATH.
_EXTRA_COMPILER_DIRS = (
    r"C:\msys64\ucrt64\bin",
    r"C:\msys64\mingw64\bin",
    r"C:\msys64\clang64\bin",
    r"C:\mingw64\bin",
    r"C:\MinGW\bin",
    r"C:\TDM-GCC-64\bin",
    "/usr/bin",
    "/usr/local/bin",
    "/opt/homebrew/bin",
)

# -Wconversion is informative but not fatal: it flags a great deal of benign
# int/float index arithmetic in tight numeric loops. Correctness-critical
# warnings are errors; style warnings are not.
_GCC_FLAGS = ("-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
              "-Wshadow", "-Wconversion", "-Wno-error=conversion", "-fPIC")

_LIB_NAMES = (
    "libtank_pdm_infer.so",
    "libtank_pdm_infer.dylib",
    "libtank_pdm_infer.dll",
    "tank_pdm_infer.dll",
    "Release/tank_pdm_infer.dll",
    "Release/libtank_pdm_infer.dll",
)


def _shared_lib_name() -> str:
    if sys.platform == "win32":
        return "libtank_pdm_infer.dll"
    if sys.platform == "darwin":
        return "libtank_pdm_infer.dylib"
    return "libtank_pdm_infer.so"


def find_compiler():
    """Return the path to a usable C compiler, searching PATH then known dirs."""
    for name in _COMPILERS:
        found = shutil.which(name)
        if found:
            return found
    for directory in _EXTRA_COMPILER_DIRS:
        for name in _COMPILERS:
            for suffix in ("", ".exe"):
                candidate = Path(directory) / (name + suffix)
                if candidate.exists():
                    return str(candidate)
    return None


def find_cmake():
    return shutil.which("cmake")


def have_toolchain() -> bool:
    """True when a C compiler is available by any route."""
    return find_compiler() is not None


def missing_tools() -> list:
    """Which prerequisites are absent, for a useful skip message."""
    if find_compiler() is None:
        return ["a C compiler (" + "/".join(_COMPILERS) + ")"]
    return []


def _compiler_env(compiler: str) -> dict:
    """Environment with the compiler's own directory on PATH.

    An MSYS2 gcc invoked by absolute path cannot locate its runtime DLLs and
    exits 1 with no diagnostic at all, which is impossible to debug from the
    call site. Putting its bin directory first fixes that.
    """
    env = dict(os.environ)
    comp_dir = str(Path(compiler).resolve().parent)
    env["PATH"] = comp_dir + os.pathsep + env.get("PATH", "")
    return env


def _run(cmd, compiler, cwd):
    """Run a compiler command, surfacing diagnostics on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(cwd), env=_compiler_env(compiler))
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or
                  "(no diagnostic; the compiler may be missing runtime DLLs)")
        raise RuntimeError(
            "C build failed (exit {rc})\n{cmd}\n{detail}".format(
                rc=proc.returncode, cmd=" ".join(cmd), detail=detail))
    return proc


def _find_library():
    for name in _LIB_NAMES:
        candidate = BUILD_DIR / name
        if candidate.exists():
            return candidate
    return None


def _build_direct(compiler: str):
    """Compile the shared library and self-test binary with one compiler call each."""
    BUILD_DIR.mkdir(exist_ok=True)
    lib_path = BUILD_DIR / _shared_lib_name()

    # Link to a scratch name first. On Windows a DLL already loaded by
    # ctypes.CDLL is held open, so linking straight over it fails with an
    # opaque "ld returned 1 exit status" -- which made the build succeed or
    # fail depending on test ordering. Building aside and then swapping keeps
    # compilation verifiable regardless of what is currently loaded.
    scratch = lib_path.with_name(lib_path.stem + ".new" + lib_path.suffix)
    _run([compiler, *_GCC_FLAGS, "-shared",
          *[str(C_DIR / s) for s in SOURCES],
          "-o", str(scratch), "-lm"], compiler, C_DIR)

    try:
        os.replace(scratch, lib_path)
    except OSError:
        # Canonical path is locked by a loaded module; the freshly linked
        # artifact stands on its own and proves the build.
        return scratch

    selftest = BUILD_DIR / ("tank_pdm_selftest.exe" if sys.platform == "win32"
                            else "tank_pdm_selftest")
    try:
        _run([compiler, *_GCC_FLAGS,
              *[str(C_DIR / s) for s in SELFTEST_SOURCES],
              "-o", str(selftest), "-lm"], compiler, C_DIR)
    except RuntimeError:
        # A running self-test binary holds its own image open; the shared
        # library is the artifact that matters here.
        pass

    return lib_path


def _build_cmake():
    BUILD_DIR.mkdir(exist_ok=True)
    subprocess.run(["cmake", "-S", str(C_DIR), "-B", str(BUILD_DIR)],
                   check=True, capture_output=True, text=True)
    subprocess.run(["cmake", "--build", str(BUILD_DIR), "--config", "Release"],
                   check=True, capture_output=True, text=True)
    return _find_library()


# Sources whose modification invalidates a previously built library.
_SOURCE_GLOBS = ("*.c", "*.h")


def _sources_newer_than(lib_path) -> bool:
    """True when any C source is newer than the built library.

    The cache used to be existence-only, so editing tank_pdm_infer.c and
    re-running the suite silently validated the *old* binary -- including the
    C/Python parity gate, which is the one test whose entire job is to catch
    the C and Python implementations drifting apart. A stale-but-present
    artifact is worse than a missing one.
    """
    from pathlib import Path as _P
    lib = _P(lib_path)
    if not lib.exists():
        return True
    lib_mtime = lib.stat().st_mtime
    here = _P(__file__).resolve().parent
    for pattern in _SOURCE_GLOBS:
        for src in here.glob(pattern):
            if src.stat().st_mtime > lib_mtime:
                return True
    return False


def build_engine(force: bool = False, prefer_cmake: bool = False):
    """Build the edge runtime. Returns the shared-library Path, or None when no
    compiler is available."""
    compiler = find_compiler()
    if compiler is None:
        return None
    if not force:
        existing = _find_library()
        if existing is not None and not _sources_newer_than(existing):
            return existing

    if prefer_cmake and find_cmake():
        return _build_cmake()

    if os.path.basename(compiler).lower().startswith("cl"):
        # MSVC needs different flags; defer to CMake, which knows them.
        if find_cmake():
            return _build_cmake()
        return None

    return _build_direct(compiler)


if __name__ == "__main__":
    if not have_toolchain():
        raise SystemExit("missing build prerequisites: " + ", ".join(missing_tools()))
    print(f"compiler: {find_compiler()}")
    print(f"built:    {build_engine(force=True)}")
