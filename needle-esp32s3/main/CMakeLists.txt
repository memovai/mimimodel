idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    REQUIRES esp_partition esp_timer
)
target_compile_options(${COMPONENT_LIB} PRIVATE -O3 -mlongcalls)
