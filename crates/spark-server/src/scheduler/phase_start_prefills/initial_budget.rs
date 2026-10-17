// SPDX-License-Identifier: AGPL-3.0-only

//! Initial issued chunk budget, separate from the larger allocation envelope.

pub(super) fn initial_chunk_budget(
    configured: usize,
    arena: usize,
    idle: bool,
    glm_c4_sparse: bool,
    requires_configured_budget: bool,
) -> usize {
    // Decode-row allocation headroom is not part of a model-required or C4
    // sparse prefill budget. Keep issued geometry independent of admission.
    if glm_c4_sparse || requires_configured_budget {
        return configured.min(arena);
    }
    if idle { arena } else { configured }
}

#[cfg(test)]
mod tests {
    use super::initial_chunk_budget;

    #[test]
    fn sparse_glm_idle_and_busy_respect_configured_budget() {
        for (configured, arena) in [(1024, 1028), (256, 260), (4, 8), (1, 5)] {
            for idle in [false, true] {
                assert_eq!(
                    initial_chunk_budget(configured, arena, idle, true, false),
                    configured,
                    "selected issued chunk must not borrow decode capacity: configured={configured}, arena={arena}, idle={idle}",
                );
            }
        }
        for idle in [false, true] {
            assert_eq!(initial_chunk_budget(1024, 512, idle, true, false), 512);
        }
    }

    #[test]
    fn ordinary_or_sparse_off_preserves_existing_solo_expansion() {
        for (configured, arena) in [(1024, 1028), (256, 260), (4, 8), (1, 5)] {
            assert_eq!(
                initial_chunk_budget(configured, arena, true, false, false),
                arena
            );
            assert_eq!(
                initial_chunk_budget(configured, arena, false, false, false),
                configured
            );
        }
    }
    #[test]
    fn native_exl3_initial_chunk_cannot_borrow_decode_headroom() {
        for idle in [true, false] {
            for sparse in [false, true] {
                assert_eq!(initial_chunk_budget(128, 129, idle, sparse, true), 128);
                assert_eq!(initial_chunk_budget(128, 64, idle, sparse, true), 64);
            }
        }
    }
}
