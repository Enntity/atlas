// SPDX-License-Identifier: AGPL-3.0-only

use crate::{Error, Result};

pub(crate) trait Fixed: Sized {
    const LEN: usize;
    fn put(&self, w: &mut Writer<'_>) -> Result<()>;
    fn get(r: &mut Reader<'_>) -> Result<Self>;
}

pub(crate) struct Writer<'a>(pub &'a mut [u8]);
impl Writer<'_> {
    pub fn bytes(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() > self.0.len() {
            return Err(Error("short output"));
        }
        let out = core::mem::take(&mut self.0);
        let (head, tail) = out.split_at_mut(bytes.len());
        head.copy_from_slice(bytes);
        self.0 = tail;
        Ok(())
    }
    pub fn field<T: Fixed>(&mut self, value: &T) -> Result<()> {
        value.put(self)
    }
}

pub(crate) struct Reader<'a>(pub &'a [u8]);
impl Reader<'_> {
    pub fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        if self.0.len() < N {
            return Err(Error("short input"));
        }
        let (head, tail) = self.0.split_at(N);
        self.0 = tail;
        head.try_into().map_err(|_| Error("short input"))
    }
    pub fn field<T: Fixed>(&mut self) -> Result<T> {
        T::get(self)
    }
}

impl<const N: usize> Fixed for [u8; N] {
    const LEN: usize = N;
    fn put(&self, w: &mut Writer<'_>) -> Result<()> {
        w.bytes(self)
    }
    fn get(r: &mut Reader<'_>) -> Result<Self> {
        r.bytes()
    }
}

macro_rules! integer {
    ($t:ty, $len:expr) => {
        impl Fixed for $t {
            const LEN: usize = $len;
            fn put(&self, w: &mut Writer<'_>) -> Result<()> {
                w.bytes(&self.to_be_bytes())
            }
            fn get(r: &mut Reader<'_>) -> Result<Self> {
                Ok(Self::from_be_bytes(r.bytes()?))
            }
        }
    };
}
integer!(u32, 4);
integer!(u64, 8);

macro_rules! record {
    ($name:ident {$($field:ident: $ty:ty),* $(,)?}) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name { $(pub $field: $ty),* }
        impl crate::codec::Fixed for $name {
            const LEN: usize = 0 $(+ <$ty as crate::codec::Fixed>::LEN)*;
            fn put(&self, w: &mut crate::codec::Writer<'_>) -> crate::Result<()> {
                $(w.field(&self.$field)?;)* Ok(())
            }
            fn get(r: &mut crate::codec::Reader<'_>) -> crate::Result<Self> {
                Ok(Self { $($field: r.field()?),* })
            }
        }
    };
}
pub(crate) use record;

macro_rules! pair {
    ($ty:ty) => {
        impl crate::codec::Fixed for [$ty; 2] {
            const LEN: usize = 2 * <$ty as crate::codec::Fixed>::LEN;
            fn put(&self, w: &mut crate::codec::Writer<'_>) -> crate::Result<()> {
                w.field(&self[0])?;
                w.field(&self[1])
            }
            fn get(r: &mut crate::codec::Reader<'_>) -> crate::Result<Self> {
                Ok([r.field()?, r.field()?])
            }
        }
    };
}
pub(crate) use pair;
