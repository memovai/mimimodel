#define NEEDLE_NO_MAIN
#include "../needle.c"
#include <assert.h>

int main(void) {
    static Needle m;
    int32_t empty_slot = -1;
    m.p2id_cap = 1; m.p2id_slot = &empty_slot; m.byte_fallback = 1;
    for (int i = 0; i < 256; i++) m.byte_id[i] = i;
    const int raw[] = {'a', 0xe2, 0x96, 0x81, 'b'};
    const int prefixed[] = {0xe2, 0x96, 0x81, 'a', 0xe2, 0x96, 0x81, 'b'};
    for (int flag = 0; flag <= 1; flag++) {
        m.add_dummy = flag;
        int out[16];
        assert(encode_raw(&m, "a b", out, 16) == 5);
        assert(memcmp(out, raw, sizeof raw) == 0);
        assert(m.add_dummy == flag);
        assert(needle_encode(&m, "a b", out, 16) == (flag ? 8 : 5));
        assert(memcmp(out, flag ? prefixed : raw, flag ? sizeof prefixed : sizeof raw) == 0);
        assert(encode_raw(&m, "a b", out, 2) == 2);
        assert(memcmp(out, raw, 2 * sizeof *raw) == 0);
        assert(encode_raw(&m, "", out, 16) == 0);
    }
    return 0;
}
