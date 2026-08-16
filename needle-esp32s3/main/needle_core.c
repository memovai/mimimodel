/* The engine lives at the repository root. This forwarding include keeps the
 * ESP-IDF build from needing a duplicate copy of needle.c, which drifted out
 * of sync more than once during development. */
#include "../../needle.c"
