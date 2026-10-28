// SPDX-License-Identifier: AGPL-3.0-only

use super::enabled;

#[test]
fn glm_mla_multiseq_is_gated() {
    if std::env::var_os("ATLAS_GLM_MLA_MULTI_SEQ").is_none() {
        assert!(!enabled());
    }
}
