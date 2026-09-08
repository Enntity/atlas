// SPDX-License-Identifier: AGPL-3.0-only
#include "glm_moe_btile_m64_bounds.h"
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
static void require(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(2); }
}
int main() {
    for (int rows : {1,2,3,4,5,15,16,17,63,64,65,127,128,129,130,148,255,256,257,1023,1024,1025,1028,1087,1088})
        for (unsigned mt=0; mt<18; ++mt) for (unsigned nt=0; nt<17; ++nt)
            require(glm_btile_m64_work_valid(rows,mt,nt)==(mt<unsigned((rows+63)/64)&&nt<16), "full-prefill M64 work coverage");
    for (int rows : {-1,0,1089}) require(!glm_btile_m64_work_valid(rows,0,0), "invalid rows rejected");
    require(!glm_btile_m64_work_valid(1088,~0u,0), "M overflow rejected");
    std::puts("PASS full-prefill M64 work coverage and bounds");
}
