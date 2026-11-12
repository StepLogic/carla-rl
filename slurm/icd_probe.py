#!/usr/bin/env python3
"""Ask the NVIDIA driver, inside a container, for vkCreateInstance. Directly.

Run in BOTH carla images. The loader's complaint --
    loader_scanned_icd_add: Could not get 'vkCreateInstance' via
    'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0
-- is loader-internal and happens during ICD scanning, so it is independent of
UE4 and of which application is running. The same driver file, mounted the same
way by the same apptainer on the same node, is accepted in the 0.9.15 image and
rejected in the 0.9.16 one. This reproduces that one step with nothing else in
the way: dlopen the driver, take vk_icdGetInstanceProcAddr, and ask it for
vkCreateInstance exactly as the loader does.

A NULL here, with the symbol present in the file (objdump says it is), means the
driver is declining; the errno and the dynamic-link state printed alongside are
then the only remaining evidence.
"""
import ctypes
import os
import sys

LIB = os.environ.get("NVLIB", "/.singularity.d/libs/libGLX_nvidia.so.0")


def main():
    print("python:", sys.version.split()[0])
    print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", "unset"))
    print("driver file:", LIB, "exists:", os.path.exists(LIB))

    try:
        h = ctypes.CDLL(LIB, mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        print("RESULT: dlopen FAILED ->", e)
        return 1
    print("dlopen: ok")

    try:
        fn = h.vk_icdGetInstanceProcAddr
    except AttributeError as e:
        print("RESULT: vk_icdGetInstanceProcAddr NOT EXPORTED ->", e)
        return 2
    print("vk_icdGetInstanceProcAddr: found")

    fn.restype = ctypes.c_void_p
    fn.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    for name in (b"vkCreateInstance", b"vkEnumerateInstanceExtensionProperties",
                 b"vkGetInstanceProcAddr", b"vkEnumerateInstanceVersion"):
        addr = fn(None, name)
        print(f"  {name.decode():42s} -> {'NULL' if not addr else hex(addr)}")

    # The legacy path the loader falls back to when the modern one yields nothing.
    try:
        print("  dlsym(vkCreateInstance) directly           ->",
              hex(ctypes.cast(h.vkCreateInstance, ctypes.c_void_p).value or 0))
    except AttributeError:
        print("  dlsym(vkCreateInstance) directly           -> NOT EXPORTED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
