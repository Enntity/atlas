// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use spark_runtime::gpu::DevicePtr;

/// Records every call it serves, as `(name, ptr or handle, bytes, stream)`.
#[derive(Default)]
struct Calls(Mutex<Vec<(&'static str, u64, usize, u64)>>);

impl Calls {
    fn push(&self, call: (&'static str, u64, usize, u64)) -> Result<()> {
        self.0.lock().push(call);
        Ok(())
    }
}

impl CommBackend for Calls {
    fn all_reduce(&self, ptr: u64, bytes: usize) -> Result<()> {
        self.push(("all_reduce", ptr, bytes, 0))
    }
    fn all_reduce_async(&self, ptr: u64, bytes: usize, stream: u64) -> Result<()> {
        self.push(("all_reduce_async", ptr, bytes, stream))
    }
    fn all_gather(&self, send: u64, _: u64, bytes: usize) -> Result<()> {
        self.push(("all_gather", send, bytes, 0))
    }
    fn reduce_scatter(&self, send: u64, _: u64, bytes: usize) -> Result<()> {
        self.push(("reduce_scatter", send, bytes, 0))
    }
    fn broadcast(&self, ptr: u64, bytes: usize, _: usize) -> Result<()> {
        self.push(("broadcast", ptr, bytes, 0))
    }
    fn barrier(&self) -> Result<()> {
        self.push(("barrier", 0, 0, 0))
    }
    fn exchange_async(
        &self,
        send: u64,
        _: u64,
        bytes: usize,
        _: bool,
        stream: u64,
    ) -> Result<bool> {
        self.push(("exchange_async", send, bytes, stream))?;
        Ok(true)
    }
    fn all_reduce_capturable(&self, ptr: u64, bytes: usize, stream: u64) -> Result<bool> {
        self.push(("all_reduce_capturable", ptr, bytes, stream))?;
        Ok(true)
    }
    fn capturable_all_reduce_max_bytes(&self) -> usize {
        4096
    }
    fn supports_exchange_async(&self, _: usize) -> bool {
        true
    }
    fn send_to(&self, ptr: u64, bytes: usize, _: usize, stream: u64) -> Result<()> {
        self.push(("send_to", ptr, bytes, stream))
    }
    fn recv_from(&self, ptr: u64, bytes: usize, _: usize, stream: u64) -> Result<()> {
        self.push(("recv_from", ptr, bytes, stream))
    }
    fn rank(&self) -> usize {
        1
    }
    fn world_size(&self) -> usize {
        2
    }
}

fn regions(base: u64) -> Vec<L2Region> {
    vec![L2Region::whole(DevicePtr(base), 4096)]
}

type Fired = Mutex<Vec<(L2Site, Vec<L2Region>, u64)>>;

fn recorder(fired: &Fired) -> impl Fn(L2Site, &[L2Region], u64) -> Result<()> + Sync + '_ {
    move |site, r: &[L2Region], stream| {
        fired.lock().push((site, r.to_vec(), stream));
        Ok(())
    }
}

#[test]
fn each_layer_forks_its_ffn_then_the_next_attention_before_its_all_reduces() {
    let (inner, fired) = (Calls::default(), Fired::default());
    let fire = recorder(&fired);
    let comm = L2AheadComm::with(&inner, &fire);
    comm.begin_layer(regions(0x1000), regions(0x2000));
    comm.all_reduce_async(0xa0, 24, 7).unwrap();
    comm.all_reduce_async(0xb0, 24, 7).unwrap();
    // A third all-reduce in a layer takes nothing.
    comm.all_reduce_async(0xc0, 24, 7).unwrap();
    // A layer with an empty FFN lead and one all-reduce.
    comm.begin_layer(Vec::new(), regions(0x3000));
    comm.all_reduce_async(0xd0, 24, 9).unwrap();
    assert_eq!(
        *fired.lock(),
        vec![
            (L2Site::Attn, regions(0x1000), 7),
            (L2Site::Ffn, regions(0x2000), 7),
        ]
    );
    let served: Vec<_> = inner.0.lock().iter().map(|c| c.1).collect();
    assert_eq!(served, vec![0xa0, 0xb0, 0xc0, 0xd0]);
}

#[test]
fn a_sync_all_reduce_counts_without_forking() {
    let (inner, fired) = (Calls::default(), Fired::default());
    let fire = recorder(&fired);
    let comm = L2AheadComm::with(&inner, &fire);
    comm.begin_layer(regions(0x1000), regions(0x2000));
    comm.all_reduce(0xa0, 24).unwrap();
    comm.all_reduce_async(0xb0, 24, 3).unwrap();
    assert_eq!(*fired.lock(), vec![(L2Site::Ffn, regions(0x2000), 3)]);
}

#[test]
fn every_other_call_is_the_inner_communicators() {
    let (inner, fired) = (Calls::default(), Fired::default());
    let fire = recorder(&fired);
    let comm = L2AheadComm::with(&inner, &fire);
    comm.begin_layer(regions(0x1000), regions(0x2000));
    comm.all_gather(1, 2, 3).unwrap();
    comm.reduce_scatter(4, 5, 6).unwrap();
    comm.broadcast(7, 8, 0).unwrap();
    comm.barrier().unwrap();
    assert!(comm.exchange_async(9, 10, 11, true, 12).unwrap());
    assert!(comm.all_reduce_capturable(13, 14, 15).unwrap());
    comm.send_to(16, 17, 0, 18).unwrap();
    comm.recv_from(19, 20, 0, 21).unwrap();
    assert_eq!(
        (
            comm.capturable_all_reduce_max_bytes(),
            comm.supports_exchange_async(1)
        ),
        (4096, true)
    );
    assert_eq!((comm.rank(), comm.world_size()), (1, 2));
    assert!(fired.lock().is_empty());
    assert_eq!(
        *inner.0.lock(),
        vec![
            ("all_gather", 1, 3, 0),
            ("reduce_scatter", 4, 6, 0),
            ("broadcast", 7, 8, 0),
            ("barrier", 0, 0, 0),
            ("exchange_async", 9, 11, 12),
            ("all_reduce_capturable", 13, 14, 15),
            ("send_to", 16, 17, 18),
            ("recv_from", 19, 20, 21),
        ]
    );
}
