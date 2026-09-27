#define NEEDLE_NO_MAIN
#include "../needle.c"
#include <assert.h>

int main(void) {
    const char *spaces[] = {"", " ", "\t\r\n "};
    const char *types[] = {"integer", "number", "string", "integerish"};
    for (unsigned i = 0; i < sizeof spaces / sizeof *spaces; i++) {
        for (unsigned j = 0; j < sizeof types / sizeof *types; j++) {
            char schema[256];
            NTool tools[1];
            snprintf(schema, sizeof schema,
                "[{\"name\":\"gpio\",\"parameters\":{\"properties\":{\"pin\":{\"type\":%s\"%s\"}}}}]",
                spaces[i], types[j]);
            assert(needle_parse_tools(schema, tools, 1) == 1);
            assert(tools[0].n_params == 1);
            assert(tools[0].params[0].is_num == (j < 2));
        }
    }
    return 0;
}
