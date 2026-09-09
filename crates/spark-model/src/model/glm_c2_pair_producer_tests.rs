// SPDX-License-Identifier: AGPL-3.0-only
//! Actual Model-issued owners; not a fabricated pair receipt or numerical proof.
use super::verdict_continuation_tests as flow;
use crate::speculative::glm_repair::GlmPairedHandoff;

#[test]
fn actual_issued_owners_reach_fixed_pair_producer() {
    if flow::isolated("pair_producer_tests::actual_issued_owners_reach_fixed_pair_producer") {
        return;
    }
    for rank in 0..2 {
        for order in [[0, 1], [1, 0]] {
            let (f, history) = flow::prepare_pair(rank, order);
            let inputs = [
                f.model.paired_input(&f.seqs[0]).unwrap(),
                f.model.paired_input(&f.seqs[1]).unwrap(),
            ];
            let tokens: [[u32; 5]; 2] =
                std::array::from_fn(|i| history[i].issued.clone().try_into().unwrap());
            let states = std::array::from_fn(|i| f.seqs[i].proposer_state.as_deref().unwrap());
            let context = f.model.glm_repair_context();
            let cap: &dyn GlmPairedHandoff = f.head.as_ref();
            for index in 0..2 {
                cap.validate_verify(&inputs[index], &tokens[index], states[index], &context)
                    .unwrap();
            }
            let before: Vec<_> = f
                .seqs
                .iter()
                .map(|s| (s.seq_len, s.tokens.clone()))
                .collect();
            f.gpu.clear();
            let result = cap.validate_verify_pair(&inputs, &tokens, states, &context);
            assert!(
                f.gpu.trace().is_empty(),
                "immutable pair check performs no GPU work"
            );
            assert_eq!(
                f.seqs
                    .iter()
                    .map(|s| (s.seq_len, s.tokens.clone()))
                    .collect::<Vec<_>>(),
                before
            );
            let facts = result.expect("actual fixed-pair producer must accept both issued owners");
            assert!(
                facts
                    .iter()
                    .all(|&(generation, attempt)| generation > 0 && attempt > 0)
            );
        }
    }
}
