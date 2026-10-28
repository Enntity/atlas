// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn the_worker_runs_only_the_shared_pieces_and_the_head_keeps_its_order() {
    use Piece::*;
    let pieces = |steps: Vec<Step>| -> Vec<Piece> {
        steps
            .into_iter()
            .filter_map(|s| match s {
                Step::Run(p) => Some(p),
                Step::Swap(_) => None,
            })
            .collect()
    };
    let g = geometry(ALL);
    assert_eq!(
        pieces(g.steps(1)),
        [GateUp(0), Down(0), GateUp(1), Down(1), Vocab]
    );
    // The unsplit order with each split projection in its place.
    assert_eq!(
        pieces(g.steps(0)),
        [
            Attention(0),
            Project(0),
            GateUp(0),
            Down(0),
            Residual(0),
            Attention(1),
            Project(1),
            GateUp(1),
            Down(1),
            Residual(1),
            Norm,
            Vocab,
            Select
        ]
    );
    assert_eq!(
        pieces(geometry(HEAD).steps(0)),
        [
            Attention(0),
            Post(0),
            Attention(1),
            Post(1),
            Norm,
            Vocab,
            Select
        ]
    );
    assert_eq!(pieces(geometry(MLP).steps(0)).last().copied(), Some(Tail));
    assert_eq!(pieces(geometry(HEAD).steps(1)), [Vocab]);
}
#[test]
fn a_shape_the_shares_cannot_serve_is_refused() {
    assert!(geometry(ALL).validate().is_ok());
    for bad in [
        Geometry {
            vocab: 31,
            ..geometry(ALL)
        },
        Geometry {
            inter: 168,
            ..geometry(ALL)
        },
        Geometry {
            hidden: 72,
            ..geometry(ALL)
        },
        Geometry {
            parts: Parts {
                mlp: false,
                head: false,
                attn: false,
            },
            ..geometry(ALL)
        },
        // The attention split rides the MLP's and needs CTA-sized shares.
        Geometry {
            parts: Parts { mlp: false, ..ATTN },
            ..geometry(ATTN)
        },
        Geometry {
            kv_dim: 48,
            ..geometry(ATTN)
        },
        Geometry {
            kv_dim: 32,
            ranks: 4,
            ..geometry(ATTN)
        },
    ] {
        assert!(bad.validate().is_err(), "{bad:?}");
    }
}
#[test]
fn swapped_halves_land_where_the_unsplit_launch_writes_whole_rows() {
    let g = geometry(ALL);
    let (r0, r1) = (Rank::new(g), Rank::new(g));
    let (p0, p1) = (Pair::new(&r0.gpu, 0), Pair::new(&r1.gpu, 1));
    // Run `swap` on both ranks, each landing what the other sent.
    let run = |swap: Swap| {
        let bytes = g.bytes(swap);
        let sent0 = r0.read(r0.split.ends(swap, 0, &Frame::serial(&r0.scratch)).0, bytes);
        let sent1 = r1.read(r1.split.ends(swap, 1, &Frame::serial(&r1.scratch)).0, bytes);
        p0.peer.borrow_mut().push_back(sent1.clone());
        p1.peer.borrow_mut().push_back(sent0.clone());
        r0.split
            .swap(swap, 0, &r0.gpu, &p0, &Frame::serial(&r0.scratch), 0)
            .unwrap();
        r1.split
            .swap(swap, 1, &r1.gpu, &p1, &Frame::serial(&r1.scratch), 0)
            .unwrap();
        (sent0, sent1)
    };

    // The head's MLP input reaches the worker's `norm_buf`.
    let hidden = g.gamma * g.hidden * 2;
    r0.fill(r0.scratch.norm_buf, hidden, 0x11);
    let (x, _) = run(Swap::Input(0));
    assert_eq!(r1.read(r1.scratch.norm_buf, hidden), x);
    assert_eq!(
        r0.read(r0.scratch.norm_buf, hidden),
        x,
        "the head keeps its own"
    );

    // Both ranks hold the whole gated activation, rank 0's half first.
    let act = g.bytes(Swap::Activation(0));
    r0.fill(r0.split.own(Swap::Activation(0), 0), act, 0x21);
    r1.fill(r1.split.own(Swap::Activation(0), 1), act, 0x22);
    let (a0, a1) = run(Swap::Activation(0));
    let whole = joined(g.gamma, g.inter, &a0, &a1);
    assert_eq!(r0.read(r0.scratch.mlp_intermediate, whole.len()), whole);
    assert_eq!(r1.read(r1.scratch.mlp_intermediate, whole.len()), whole);

    // The head holds the whole down projection; the worker's residual stream
    // is not its to write.
    let out = g.bytes(Swap::Output(0));
    r0.fill(r0.split.own(Swap::Output(0), 0), out, 0x31);
    r1.fill(r1.split.own(Swap::Output(0), 1), out, 0x32);
    r1.fill(r1.scratch.stream_acc, hidden, 0x3f);
    let (y0, y1) = run(Swap::Output(0));
    assert_eq!(
        r0.read(r0.scratch.stream_acc, hidden),
        joined(g.gamma, g.hidden, &y0, &y1)
    );
    assert_eq!(r1.read(r1.scratch.stream_acc, hidden).len(), hidden);
    let untouched: Vec<u8> = (0..hidden)
        .map(|i| 0x3f ^ (i as u8).wrapping_mul(31))
        .collect();
    assert_eq!(r1.read(r1.scratch.stream_acc, hidden), untouched);

    // The final norm rows reach the worker; the head gets the whole logits.
    r0.fill(r0.scratch.norm_buf, hidden, 0x41);
    let (h, _) = run(Swap::Hidden);
    assert_eq!(r1.read(r1.scratch.norm_buf, hidden), h);
    let half_logits = g.bytes(Swap::Logits);
    r0.fill(r0.split.own(Swap::Logits, 0), half_logits, 0x51);
    r1.fill(r1.split.own(Swap::Logits, 1), half_logits, 0x52);
    let (l0, l1) = run(Swap::Logits);
    assert_eq!(
        r0.read(r0.scratch.logits, g.gamma * g.vocab * 2),
        joined(g.gamma, g.vocab, &l0, &l1)
    );

    // Same sizes, same order, on both ranks.
    assert_eq!(*p0.sizes.borrow(), *p1.sizes.borrow());
}
#[test]
fn the_chain_gathers_shares_where_the_unsplit_launch_writes_whole_rows() {
    let g = Geometry {
        ranks: 4,
        ..geometry(ALL)
    };
    assert!(g.validate().is_ok());
    let ranks: Vec<Rank> = (0..4).map(|_| Rank::new(g)).collect();
    let gpus = [&ranks[0].gpu, &ranks[1].gpu, &ranks[2].gpu, &ranks[3].gpu];
    let head_rows = RefCell::new(Vec::new());
    let comms: Vec<Chain> = (0..4)
        .map(|rank| Chain {
            gpus,
            rank,
            head_rows: &head_rows,
            sizes: RefCell::default(),
        })
        .collect();
    let run = |swap: Swap| {
        for (rank, (r, c)) in ranks.iter().zip(&comms).enumerate() {
            r.split
                .swap(swap, rank, &r.gpu, c, &Frame::serial(&r.scratch), 0)
                .unwrap();
        }
    };
    // Every rank's share of `swap`, distinct bytes a rank.
    let fill = |swap: Swap, tag: u8| -> Vec<Vec<u8>> {
        let bytes = g.bytes(swap);
        (0..4)
            .map(|k| {
                let r = &ranks[k];
                r.fill(r.split.own(swap, k), bytes, tag + k as u8);
                r.read(r.split.own(swap, k), bytes)
            })
            .collect()
    };

    // The head's MLP input reaches every worker's `norm_buf`.
    let hidden = g.gamma * g.hidden * 2;
    ranks[0].fill(ranks[0].scratch.norm_buf, hidden, 0x11);
    let x = ranks[0].read(ranks[0].scratch.norm_buf, hidden);
    run(Swap::Input(0));
    for r in &ranks {
        assert_eq!(r.read(r.scratch.norm_buf, hidden), x);
    }

    // Every rank holds the whole gated activation, the shares in rank order.
    let parts = fill(Swap::Activation(0), 0x20);
    run(Swap::Activation(0));
    let whole = joined_shares(g.gamma, g.inter, &parts);
    for r in &ranks {
        assert_eq!(r.read(r.scratch.mlp_intermediate, whole.len()), whole);
    }

    // The head holds the whole down projection and the whole logits.
    let parts = fill(Swap::Output(0), 0x30);
    run(Swap::Output(0));
    assert_eq!(
        ranks[0].read(ranks[0].scratch.stream_acc, hidden),
        joined_shares(g.gamma, g.hidden, &parts)
    );
    let parts = fill(Swap::Logits, 0x50);
    run(Swap::Logits);
    let vocab = g.gamma * g.vocab * 2;
    assert_eq!(
        ranks[0].read(ranks[0].scratch.logits, vocab),
        joined_shares(g.gamma, g.vocab, &parts)
    );

    // Same sizes, same order, on every rank; a stopped propose drains the
    // rest at the planned sizes.
    for c in &comms[1..] {
        assert_eq!(*c.sizes.borrow(), *comms[0].sizes.borrow());
    }
    let rank = &ranks[1];
    let chain = &comms[1];
    chain.sizes.borrow_mut().clear();
    rank.split.begin(g.gamma, CtxRows::default()).unwrap();
    rank.split.finish(chain, 0).unwrap();
    let planned: Vec<usize> = g.swaps().iter().map(|&s| g.bytes(s)).collect();
    assert_eq!(*chain.sizes.borrow(), planned);
}
#[test]
fn the_attention_split_names_its_part_and_plans_its_swaps() {
    use Piece::*;
    assert_eq!(Parts::parse(Some("mlp,head,attn")).unwrap(), Some(ATTN));
    // `1` keeps its meaning: the MLP and the head.
    assert_eq!(Parts::parse(Some("1")).unwrap(), Some(ALL));
    assert_eq!(Parts::word(Some(ATTN)), 3 | 1 << 5);
    for ranks in [2, 4] {
        let g = Geometry {
            ranks,
            ..geometry(ATTN)
        };
        assert!(g.validate().is_ok());
        let (head, worker) = (g.steps(0), g.steps(ranks - 1));
        assert_eq!(swaps_of(&head), swaps_of(&worker));
        assert_eq!(g.swaps().len(), 7 * g.layers + 2);
        let pieces = |steps: Vec<Step>| -> Vec<Piece> {
            steps
                .into_iter()
                .filter_map(|s| match s {
                    Step::Run(p) => Some(p),
                    Step::Swap(_) => None,
                })
                .collect()
        };
        assert_eq!(
            pieces(worker),
            [
                Qkv(0),
                OProj(0),
                GateUp(0),
                Down(0),
                Qkv(1),
                OProj(1),
                GateUp(1),
                Down(1),
                Vocab
            ]
        );
        assert_eq!(
            pieces(head)[..8],
            [
                AttnPre(0),
                Qkv(0),
                AttnCore(0),
                OProj(0),
                ProjectRest(0),
                GateUp(0),
                Down(0),
                Residual(0)
            ]
        );
    }
    let g = geometry(ATTN);
    // q's 64-row share plus k's and v's 32 a rank, at gamma rows.
    assert_eq!(g.bytes(Swap::Qkv(0)), 3 * (64 + 32 + 32) * 2);
    assert_eq!(g.bytes(Swap::AttnIn(0)), 3 * 64 * 2);
    assert_eq!(g.bytes(Swap::AttnOut(0)), 3 * 128 * 2);
    assert_eq!(g.bytes(Swap::OProj(0)), 3 * 32 * 2);
}
#[test]
fn the_chain_joins_the_attention_shares_where_the_unsplit_projections_write() {
    let g = Geometry {
        ranks: 4,
        ..geometry(ATTN)
    };
    let ranks: Vec<Rank> = (0..4).map(|_| Rank::new(g)).collect();
    let gpus = [&ranks[0].gpu, &ranks[1].gpu, &ranks[2].gpu, &ranks[3].gpu];
    let head_rows = RefCell::new(Vec::new());
    let comms: Vec<Chain> = (0..4)
        .map(|rank| Chain {
            gpus,
            rank,
            head_rows: &head_rows,
            sizes: RefCell::default(),
        })
        .collect();
    let run = |swap: Swap| {
        for (rank, (r, c)) in ranks.iter().zip(&comms).enumerate() {
            r.split
                .swap(swap, rank, &r.gpu, c, &Frame::serial(&r.scratch), 0)
                .unwrap();
        }
    };
    for r in &ranks {
        r.split.begin(g.gamma, CtxRows::default()).unwrap();
    }
    // The normed input and the attention rows reach every worker.
    let hidden = g.gamma * g.hidden * 2;
    ranks[0].fill(ranks[0].scratch.norm_buf, hidden, 0x11);
    run(Swap::AttnIn(0));
    let x = ranks[0].read(ranks[0].scratch.norm_buf, hidden);
    for r in &ranks[1..] {
        assert_eq!(r.read(r.scratch.norm_buf, hidden), x);
    }
    let attn = g.gamma * g.q_dim * 2;
    ranks[0].fill(ranks[0].scratch.attn_out, attn, 0x12);
    run(Swap::AttnOut(0));
    let y = ranks[0].read(ranks[0].scratch.attn_out, attn);
    for r in &ranks[1..] {
        assert_eq!(r.read(r.scratch.attn_out, attn), y);
    }
    // Every rank's q, k and v shares, joined on the head into whole rows.
    let bytes = g.bytes(Swap::Qkv(0));
    let parts: Vec<Vec<u8>> = (0..4)
        .map(|k| {
            let r = &ranks[k];
            r.fill(r.split.own(Swap::Qkv(0), k), bytes, 0x40 + k as u8);
            r.read(r.split.own(Swap::Qkv(0), k), bytes)
        })
        .collect();
    run(Swap::Qkv(0));
    let (wq, wk) = (widest(g.q_dim, 4), widest(g.kv_dim, 4));
    for (at, dst, n) in [
        (0, ranks[0].scratch.q_buf, g.q_dim),
        (wq, ranks[0].scratch.k_buf, g.kv_dim),
        (wq + wk, ranks[0].scratch.v_buf, g.kv_dim),
    ] {
        let subs: Vec<Vec<u8>> = parts
            .iter()
            .map(|p| p[at * g.gamma * 2..].to_vec())
            .collect();
        assert_eq!(
            ranks[0].read(dst, g.gamma * n * 2),
            joined_shares(g.gamma, n, &subs),
            "part at {at}"
        );
    }
    // And o_proj's shares, into the rows o_proj writes.
    let out = g.bytes(Swap::OProj(0));
    let parts: Vec<Vec<u8>> = (0..4)
        .map(|k| {
            let r = &ranks[k];
            r.fill(r.split.own(Swap::OProj(0), k), out, 0x60 + k as u8);
            r.read(r.split.own(Swap::OProj(0), k), out)
        })
        .collect();
    run(Swap::OProj(0));
    assert_eq!(
        ranks[0].read(ranks[0].scratch.stream_acc, hidden),
        joined_shares(g.gamma, g.hidden, &parts)
    );
    for c in &comms[1..] {
        assert_eq!(*c.sizes.borrow(), *comms[0].sizes.borrow());
    }
}
