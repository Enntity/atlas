// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;

/// A `serde_json` formatter that matches Python `json.dumps` default
/// separators: `", "` between array/object items and `": "` between an
/// object key and its value. serde_json's `CompactFormatter` (the
/// default) emits no spaces, while jinja2's `tojson` uses `json.dumps`
/// defaults. This bridges the two so Atlas's `<tools>` prompt block is
/// byte-identical to transformers/vLLM.
#[derive(Clone, Debug)]
struct PythonJsonFormatter;

pub(super) fn python_tojson(value: minijinja::Value) -> Result<minijinja::Value, minijinja::Error> {
    let mut buf = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut buf, PythonJsonFormatter);
    serde::Serialize::serialize(&value, &mut serializer).map_err(|error| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            format!("tojson serialization failed: {error}"),
        )
    })?;
    let json = String::from_utf8(buf).map_err(|error| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            format!("tojson produced invalid UTF-8: {error}"),
        )
    })?;
    Ok(minijinja::Value::from_safe_string(json))
}

impl serde_json::ser::Formatter for PythonJsonFormatter {
    #[inline]
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    #[inline]
    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    #[inline]
    fn begin_object_value<W>(&mut self, writer: &mut W) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        writer.write_all(b": ")
    }
}
