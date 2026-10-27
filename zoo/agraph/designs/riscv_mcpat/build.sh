#!/bin/sh
# Build the static x86 GEMM kernel against gem5's libm5.a.
# GEM5_PATH is the gem5 source root, with libm5.a built by: cd $GEM5_PATH/util/m5 && scons build/x86/out/m5
set -e
: "${GEM5_PATH:?set GEM5_PATH to the gem5 source root}"
dir=$(dirname "$0")
gcc -static -O2 -I "$GEM5_PATH/include" -o "$dir/gemm" "$dir/gemm.c" "$GEM5_PATH/util/m5/build/x86/out/libm5.a"
