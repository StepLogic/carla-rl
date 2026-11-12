#!/bin/sh
# Runs INSIDE the carla0916 container. A real file, not an inline heredoc: the
# previous attempt's nested quoting broke the `for f in $(find ...)` loop and
# produced "Syntax error: end of file unexpected" instead of UE4's log.
#
# Prints, in order: what the Vulkan loader decides, what the UE4 binary says on
# stderr, and whatever UE4 wrote to its own log. The loader trace is the point --
# every attempt so far has had CARLA die silently, and VK_LOADER_DEBUG is the one
# thing that explains a silent Vulkan death.
set -u
PORT=${PORT:-12000}
echo "### loader and ICD state"
echo "libvulkan resolves to: $(ls -l /usr/lib/x86_64-linux-gnu/libvulkan.so.1 2>/dev/null)"
md5sum /usr/lib/x86_64-linux-gnu/libvulkan.so.1* 2>/dev/null
echo "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-unset}"
ls -la /etc/vulkan/icd.d/ /usr/share/vulkan/icd.d/ 2>/dev/null

echo "### loader trace while enumerating devices (vulkan via python/ctypes)"
VK_LOADER_DEBUG=all python3 - 2>&1 <<'PY' | tail -40
import ctypes, ctypes.util
lib = ctypes.CDLL("libvulkan.so.1")
# vkEnumerateInstanceVersion exists from 1.1; print what the loader reports.
try:
    ver = ctypes.c_uint32()
    if lib.vkEnumerateInstanceVersion(ctypes.byref(ver)) == 0:
        v = ver.value
        print("loader instance version: %d.%d.%d" % (v >> 22, (v >> 12) & 0x3FF, v & 0xFFF))
except AttributeError:
    print("no vkEnumerateInstanceVersion (loader older than 1.1)")
PY

echo "### UE4 binary, direct, with the loader talking"
BIN=/workspace/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping
[ -x "$BIN" ] || BIN=$(find /workspace -name 'CarlaUE4-Linux-Shipping' 2>/dev/null | head -1)
echo "binary: $BIN"
VK_LOADER_DEBUG=${VKDBG:-all} timeout 60 "$BIN" CarlaUE4 \
  -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port="$PORT" 2>&1 \
  | grep -iE "icd|nvidia|lvp|lavapipe|driver|device|instance|error|cannot|fail" | head -60
echo "### exit: $?"

echo "### UE4's own logs"
for d in /workspace/CarlaUE4/Saved/Logs "$HOME/.config/Epic/CarlaUE4/Saved/Logs"; do
  [ -d "$d" ] || continue
  echo "--- $d"; ls -la "$d" 2>/dev/null | head
  for f in "$d"/*.log; do
    [ -f "$f" ] || continue
    echo "=== $f"; grep -iE "vulkan|rhi|device|adapter|error|fatal|gpu" "$f" | tail -25
  done
done
echo "### inner done"
