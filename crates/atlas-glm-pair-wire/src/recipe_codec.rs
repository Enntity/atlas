// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::Error;

struct Writer(Vec<u8>);
impl Writer {
    fn bytes(&mut self, b: &[u8]) -> Result<()> {
        if self
            .0
            .len()
            .checked_add(b.len())
            .is_none_or(|n| n > MAX_RECIPE_BYTES)
        {
            return Err(Error("recipe exceeds 64 KiB"));
        }
        self.0.extend_from_slice(b);
        Ok(())
    }
    fn string(&mut self, s: &str) -> Result<()> {
        self.u32(u32::try_from(s.len()).map_err(|_| Error("recipe string length"))?)?;
        self.bytes(s.as_bytes())
    }
    fn boolean(&mut self, v: bool) -> Result<()> {
        self.u8(u8::from(v))
    }
    fn vector<T>(
        &mut self,
        values: &[T],
        mut put: impl FnMut(&mut Self, &T) -> Result<()>,
    ) -> Result<()> {
        self.u16(u16::try_from(values.len()).map_err(|_| Error("recipe vector count"))?)?;
        for v in values {
            put(self, v)?;
        }
        Ok(())
    }
    fn strings(&mut self, v: &[String]) -> Result<()> {
        self.vector(v, |w, s| w.string(s))
    }
    fn pairs(&mut self, v: &[(String, String)]) -> Result<()> {
        self.vector(v, |w, (k, v)| {
            w.string(k)?;
            w.string(v)
        })
    }
}

struct Reader<'a>(&'a [u8]);
impl<'a> Reader<'a> {
    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.0.len() {
            return Err(Error("truncated recipe"));
        }
        let (v, tail) = self.0.split_at(n);
        self.0 = tail;
        Ok(v)
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u32()? as usize;
        if n > 4096 {
            return Err(Error("recipe string exceeds 4096 bytes"));
        }
        let v =
            std::str::from_utf8(self.bytes(n)?).map_err(|_| Error("recipe string is not UTF-8"))?;
        if v.contains('\0') {
            return Err(Error("NUL in recipe string"));
        }
        Ok(v.to_owned())
    }
    fn boolean(&mut self) -> Result<bool> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(Error("noncanonical recipe bool")),
        }
    }
    fn vector<T>(
        &mut self,
        max: usize,
        mut get: impl FnMut(&mut Self) -> Result<T>,
    ) -> Result<Vec<T>> {
        let n = usize::from(self.u16()?);
        if n > max {
            return Err(Error("recipe vector exceeds bound"));
        }
        let mut values = Vec::with_capacity(n);
        for _ in 0..n {
            values.push(get(self)?);
        }
        Ok(values)
    }
    fn strings(&mut self, max: usize) -> Result<Vec<String>> {
        self.vector(max, Self::string)
    }
    fn pairs(&mut self, max: usize) -> Result<Vec<(String, String)>> {
        self.vector(max, |r| Ok((r.string()?, r.string()?)))
    }
    fn digest(&mut self) -> Result<Digest> {
        self.bytes(32)?
            .try_into()
            .map_err(|_| Error("truncated recipe digest"))
    }
}

macro_rules! number {
    ($name:ident,$ty:ty,$len:expr) => {
        impl Writer {
            fn $name(&mut self, v: $ty) -> Result<()> {
                self.bytes(&v.to_be_bytes())
            }
        }
        impl Reader<'_> {
            fn $name(&mut self) -> Result<$ty> {
                Ok(<$ty>::from_be_bytes(
                    self.bytes($len)?
                        .try_into()
                        .map_err(|_| Error("truncated number"))?,
                ))
            }
        }
    };
}
number!(u8, u8, 1);
number!(u16, u16, 2);
number!(u32, u32, 4);
number!(u64, u64, 8);
number!(i64, i64, 8);

pub(super) fn encode(v: &Recipe) -> Result<Vec<u8>> {
    let mut w = Writer(Vec::new());
    w.u16(1)?;
    w.strings(&v.argv)?;
    w.pairs(&v.environment)?;
    w.vector(&v.mounts, |w, m| {
        w.string(&m.source)?;
        w.string(&m.destination)?;
        w.boolean(m.read_only)?;
        w.string(&m.propagation)
    })?;
    w.bytes(&v.image_digest)?;
    w.bytes(&v.guard_elf_digest)?;
    w.bytes(&v.server_elf_digest)?;
    w.u8(v.rank)?;
    w.u8(v.world)?;
    put_profile(&mut w, &v.profile)?;
    put_resources(&mut w, &v.resources)?;
    Ok(w.0)
}

pub(super) fn decode(bytes: &[u8]) -> Result<Recipe> {
    if bytes.len() > MAX_RECIPE_BYTES {
        return Err(Error("recipe exceeds 64 KiB"));
    }
    let mut r = Reader(bytes);
    if r.u16()? != 1 {
        return Err(Error("unsupported recipe version"));
    }
    let value = Recipe {
        argv: r.strings(64)?,
        environment: r.pairs(128)?,
        mounts: r.vector(32, |r| {
            Ok(Mount {
                source: r.string()?,
                destination: r.string()?,
                read_only: r.boolean()?,
                propagation: r.string()?,
            })
        })?,
        image_digest: r.digest()?,
        guard_elf_digest: r.digest()?,
        server_elf_digest: r.digest()?,
        rank: r.u8()?,
        world: r.u8()?,
        profile: get_profile(&mut r)?,
        resources: get_resources(&mut r)?,
    };
    if !r.0.is_empty() {
        return Err(Error("trailing recipe bytes"));
    }
    Ok(value)
}

fn put_profile(w: &mut Writer, p: &Profile) -> Result<()> {
    w.u8(p.tp)?;
    w.u8(p.ep)?;
    w.u8(p.ep_protocol)?;
    w.u16(p.max_sequences)?;
    w.u32(p.context)?;
    w.u32(p.prefill)?;
    w.u8(p.drafts)?;
    w.boolean(p.eager)?;
    w.u8(p.kv_format)?;
    w.u32(p.cold_min)?;
    w.u32(p.cold_max)
}
fn get_profile(r: &mut Reader<'_>) -> Result<Profile> {
    Ok(Profile {
        tp: r.u8()?,
        ep: r.u8()?,
        ep_protocol: r.u8()?,
        max_sequences: r.u16()?,
        context: r.u32()?,
        prefill: r.u32()?,
        drafts: r.u8()?,
        eager: r.boolean()?,
        kv_format: r.u8()?,
        cold_min: r.u32()?,
        cold_max: r.u32()?,
    })
}
fn put_resources(w: &mut Writer, v: &Resources) -> Result<()> {
    w.u64(v.memory)?;
    w.u64(v.swap)?;
    w.string(&v.cpuset)?;
    w.u64(v.shm)?;
    w.vector(&v.device_requests, |w, d| {
        w.string(&d.driver)?;
        w.i64(d.count)?;
        w.strings(&d.device_ids)?;
        w.vector(&d.capabilities, |w, c| w.strings(c))?;
        w.pairs(&d.options)
    })?;
    w.vector(&v.ulimits, |w, u| {
        w.string(&u.name)?;
        w.i64(u.soft)?;
        w.i64(u.hard)
    })?;
    w.strings(&v.cap_add)?;
    w.strings(&v.cap_drop)?;
    w.u32(v.uid)?;
    w.u32(v.gid)?;
    w.string(&v.network_mode)?;
    w.string(&v.pid_mode)?;
    w.string(&v.restart_policy)?;
    w.boolean(v.init)?;
    w.boolean(v.no_new_privileges)?;
    w.string(&v.ipc_mode)
}
fn get_resources(r: &mut Reader<'_>) -> Result<Resources> {
    Ok(Resources {
        memory: r.u64()?,
        swap: r.u64()?,
        cpuset: r.string()?,
        shm: r.u64()?,
        device_requests: r.vector(8, |r| {
            Ok(DeviceRequest {
                driver: r.string()?,
                count: r.i64()?,
                device_ids: r.strings(8)?,
                capabilities: r.vector(8, |r| r.strings(16))?,
                options: r.pairs(32)?,
            })
        })?,
        ulimits: r.vector(32, |r| {
            Ok(Ulimit {
                name: r.string()?,
                soft: r.i64()?,
                hard: r.i64()?,
            })
        })?,
        cap_add: r.strings(64)?,
        cap_drop: r.strings(64)?,
        uid: r.u32()?,
        gid: r.u32()?,
        network_mode: r.string()?,
        pid_mode: r.string()?,
        restart_policy: r.string()?,
        init: r.boolean()?,
        no_new_privileges: r.boolean()?,
        ipc_mode: r.string()?,
    })
}
