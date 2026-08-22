idf_component_register(
    SRCS "main.c" "cq2_dot_tie728.S"
    INCLUDE_DIRS "."
    REQUIRES esp_partition esp_timer
    PRIV_REQUIRES perfmon
)
option(NEEDLE_FAST_MATH "Use faster non-IEEE math; validate output parity" ON)
option(NEEDLE_PROFILE "Collect per-phase inference timings" OFF)
option(NEEDLE_BYTE_GRAMMAR "Use continuous schema-constrained byte decoding" ON)
set(NEEDLE_PREFIX_SINK_TOKENS 160 CACHE STRING
    "Protected prompt-prefix tokens kept alongside the 256-token recent window")
set(NEEDLE_REASON_MAX_TOKENS 256 CACHE STRING
    "Maximum unconstrained reasoning tokens before a tool call")

target_compile_options(${COMPONENT_LIB} PRIVATE -O3 -mlongcalls)
target_compile_definitions(${COMPONENT_LIB} PRIVATE
    NEEDLE_PREFIX_SINK_DEFAULT=${NEEDLE_PREFIX_SINK_TOKENS}
    NEEDLE_REASON_MAX_DEFAULT=${NEEDLE_REASON_MAX_TOKENS}
)
if(NEEDLE_FAST_MATH)
    target_compile_options(${COMPONENT_LIB} PRIVATE -ffast-math)
endif()
if(NEEDLE_BYTE_GRAMMAR)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE NEEDLE_BYTE_GRAMMAR_DEFAULT=1)
else()
    target_compile_definitions(${COMPONENT_LIB} PRIVATE NEEDLE_BYTE_GRAMMAR_DEFAULT=0)
endif()
if(NEEDLE_PROFILE)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE NEEDLE_PROF)
endif()
