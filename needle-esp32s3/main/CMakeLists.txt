idf_component_register(
    SRCS "main.c" "cq2_dot_tie728.S"
    INCLUDE_DIRS "."
    REQUIRES esp_partition esp_timer
    PRIV_REQUIRES perfmon
)
option(NEEDLE_FAST_MATH "Use faster non-IEEE math; validate output parity" OFF)
option(NEEDLE_PROFILE "Collect per-phase inference timings" OFF)

target_compile_options(${COMPONENT_LIB} PRIVATE -O3 -mlongcalls)
if(NEEDLE_FAST_MATH)
    target_compile_options(${COMPONENT_LIB} PRIVATE -ffast-math)
endif()
if(NEEDLE_PROFILE)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE NEEDLE_PROF)
endif()
