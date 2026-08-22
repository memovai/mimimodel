#define NEEDLE_NO_MAIN
#include "../needle.c"

#include <assert.h>

static ByteGrammar grammar_for(NTool *tools, int n_tools) {
    ByteGrammar grammar;
    memset(&grammar, 0, sizeof grammar);
    grammar.state = BG_ARRAY_OPEN;
    grammar.tools = tools;
    grammar.n_tools = n_tools;
    grammar.max_calls = 4;
    grammar.tool = grammar.param = -1;
    return grammar;
}

static int consume_text(ByteGrammar *grammar, const char *text) {
    for (const unsigned char *p = (const unsigned char *)text; *p; p++)
        if (!bg_consume_byte(grammar, *p)) return 0;
    return 1;
}

int main(void) {
    NTool tools[2] = {0};
    strcpy(tools[0].name, "gpio_on");
    strcpy(tools[0].params[0].name, "pin");
    tools[0].params[0].is_num = 1;
    tools[0].params[0].required = 1;
    tools[0].n_params = 1;
    strcpy(tools[1].name, "notify");
    strcpy(tools[1].params[0].name, "message");
    tools[1].params[0].required = 1;
    tools[1].n_params = 1;

    ByteGrammar valid = grammar_for(tools, 2);
    assert(consume_text(&valid,
        "[{\"name\":\"gpio_on\",\"arguments\":{\"pin\":5}},"
        "{\"name\":\"notify\",\"arguments\":{\"message\":\"done\"}}]"));
    assert(valid.state == BG_DONE);

    ByteGrammar missing_required = grammar_for(tools, 2);
    assert(!consume_text(&missing_required,
        "[{\"name\":\"gpio_on\",\"arguments\":{}}]"));

    ByteGrammar repeated_tool = grammar_for(tools, 2);
    assert(!consume_text(&repeated_tool,
        "[{\"name\":\"gpio_on\",\"arguments\":{\"pin\":5}},"
        "{\"name\":\"gpio_on\",\"arguments\":{\"pin\":6}}]"));
    return 0;
}
