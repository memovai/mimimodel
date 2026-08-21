/* Two CQ2 rows against one float activation block.
 *
 * TIE728 cannot gather codebook entries, so packed indices stay scalar. The
 * useful part is its 128-bit float load: each aligned codebook entry and four
 * activations arrive in one instruction. Four independent accumulators per
 * row hide FPU madd latency, following ESP-DL's float dot-product schedule.
 *
 * void needle_cq2_dot2_tie728(float out[2], const uint8_t *row0,
 *     const uint8_t *row1, const float *x, const float *lut, int nbytes);
 */
    .text
    .align 4
    .global needle_cq2_dot2_tie728
    .type needle_cq2_dot2_tie728, @function
needle_cq2_dot2_tie728:
    entry sp, 16

    movi.n a8, 0
    wfr f0, a8
    wfr f1, a8
    wfr f2, a8
    wfr f3, a8
    wfr f4, a8
    wfr f5, a8
    wfr f6, a8
    wfr f7, a8

    loopnez a7, .L_cq2_dot2_end
        l8ui a8, a3, 0
        l8ui a9, a4, 0
        addi.n a3, a3, 1
        addi.n a4, a4, 1
        slli a8, a8, 4
        slli a9, a9, 4
        add a8, a8, a6
        add a9, a9, a6

        ee.ldf.128.ip f11, f10, f9, f8, a5, 16
        ee.ldf.128.ip f15, f14, f13, f12, a8, 0
        madd.s f0, f12, f8
        madd.s f1, f13, f9
        madd.s f2, f14, f10
        madd.s f3, f15, f11

        ee.ldf.128.ip f15, f14, f13, f12, a9, 0
        madd.s f4, f12, f8
        madd.s f5, f13, f9
        madd.s f6, f14, f10
        madd.s f7, f15, f11
    .L_cq2_dot2_end:

    add.s f0, f0, f1
    add.s f2, f2, f3
    add.s f0, f0, f2
    add.s f4, f4, f5
    add.s f6, f6, f7
    add.s f4, f4, f6
    ssi f0, a2, 0
    ssi f4, a2, 4
    retw

    .size needle_cq2_dot2_tie728, .-needle_cq2_dot2_tie728
