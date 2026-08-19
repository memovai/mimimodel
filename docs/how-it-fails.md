# How it fails

[Back to the main README](../README.md)

Cactus publishes both an [inference engine](https://github.com/cactus-compute/cactus) and the
[Needle model](https://github.com/cactus-compute/needle). ESP32-S3 appears as a supported target,
so cross-compiling the engine is the obvious first approach. The source exposes two blockers.

## 1. The Needle compute core ships as a binary

The open engine contains `ModelType::NEEDLE` and the prompt-formatting path. Its open source does
not contain Needle's layer implementation: searching the `cactus` repository for `engram`, a core
part of the architecture, returns no implementation. The compute path lives in prebuilt,
platform-specific libraries distributed through Hugging Face.

This means a new target needs the missing compute core as well as a platform build. A compiler
cannot reconstruct those layers from the public engine sources.

## 2. The open kernels target ARM NEON

All 15 open kernel source files include `arm_neon.h`. Together they use roughly 125 NEON
intrinsics and provide no scalar fallback. The same code also uses `_Float16` and `__fp16` around
770 times; Xtensa GCC does not provide those ARM-oriented types in this build path.

Replacing these kernels requires a new scalar or Xtensa implementation, plus numerical validation
for every operator. A target flag alone cannot bridge the ISA and datatype gap.

## Why a dedicated engine is still possible

The model repository provides the pieces required for an independent implementation:

- `needle/needle/model/` defines the architecture and decode loop.
- The quantizer describes how packed weights are reconstructed.
- `export.py` documents the `.cact` format at byte level.
- The exported file carries the architecture geometry and tensors needed at runtime.

This project reads that specification directly and implements the required operators in a small C
runtime for ESP32-S3. The result avoids dependence on the missing binary compute core and replaces
the ARM kernel layer with code designed for the device's memory and instruction constraints.
