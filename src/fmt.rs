//! `printf`-style formatting, backed by `core::fmt`.
//!
//! Rust has no variadics, so `printf` takes a format string plus a
//! slice of [`PrintfArg`]s (built with [`pargs!`] or [`IntoPrintfArg`]):
//!
//! ```ignore
//! let s = format_printf("pid=%d name=%s hex=%08x", pargs![42, "init", 255]);
//! assert_eq!(s, "pid=42 name=init hex=000000ff");
//! ```
//!
//! Supported conversions: `%d`/`%i` (signed), `%u` (unsigned),
//! `%x`/`%X` (hex), `%s` (string), `%c` (char), `%%` (literal
//! percent). Width (`%5d`), left-align (`%-5d`) and zero-pad
//! (`%05d`, `%08x`) are supported. Each argument is first rendered
//! with `core::fmt` (`Display`/`LowerHex`/`UpperHex`) and only the
//! padding is done here — this is not a second formatter. Unknown
//! conversions are written out literally (`%q` stays `%q`).

use alloc::string::String;

/// One `printf` argument.
#[derive(Clone, Copy, Debug)]
pub enum PrintfArg<'a> {
    Int(i64),
    UInt(u64),
    Str(&'a str),
    Char(char),
}

/// Convert a Rust value into a [`PrintfArg`]. Signed integers become
/// [`PrintfArg::Int`], unsigned become [`PrintfArg::UInt`].
pub trait IntoPrintfArg<'a> {
    fn into_printf_arg(self) -> PrintfArg<'a>;
}

impl<'a> IntoPrintfArg<'a> for i32 {
    fn into_printf_arg(self) -> PrintfArg<'a> {
        PrintfArg::Int(self as i64)
    }
}
impl<'a> IntoPrintfArg<'a> for i64 {
    fn into_printf_arg(self) -> PrintfArg<'a> {
        PrintfArg::Int(self)
    }
}
impl<'a> IntoPrintfArg<'a> for isize {
    fn into_printf_arg(self) -> PrintfArg<'a> {
        PrintfArg::Int(self as i64)
    }
}
impl<'a> IntoPrintfArg<'a> for u32 {
    fn into_printf_arg(self) -> PrintfArg<'a> {
        PrintfArg::UInt(self as u64)
    }
}
impl<'a> IntoPrintfArg<'a> for u64 {
    fn into_printf_arg(self) -> PrintfArg<'a> {
        PrintfArg::UInt(self)
    }
}
impl<'a> IntoPrintfArg<'a> for usize {
    fn into_printf_arg(self) -> PrintfArg<'a> {
        PrintfArg::UInt(self as u64)
    }
}
impl<'a> IntoPrintfArg<'a> for &'a str {
    fn into_printf_arg(self) -> PrintfArg<'a> {
        PrintfArg::Str(self)
    }
}
impl<'a> IntoPrintfArg<'a> for char {
    fn into_printf_arg(self) -> PrintfArg<'a> {
        PrintfArg::Char(self)
    }
}

/// Build a `&[PrintfArg]` slice from values:
/// `pargs![42, "init", 'x']`.
#[macro_export]
macro_rules! pargs {
    ($($x:expr),* $(,)?) => {
        &[$($crate::fmt::IntoPrintfArg::into_printf_arg($x)),*] as &[ $crate::fmt::PrintfArg<'_> ]
    };
}

struct Spec {
    left_align: bool,
    zero_pad: bool,
    width: usize,
    conv: char,
}

/// Parse one `%...` conversion starting after the `%`.
/// Returns `(spec, bytes_consumed)`.
fn parse_spec(bytes: &[u8]) -> (Spec, usize) {
    let mut i = 0;
    let mut left_align = false;
    let mut zero_pad = false;
    if i < bytes.len() && bytes[i] == b'-' {
        left_align = true;
        i += 1;
    }
    if i < bytes.len() && bytes[i] == b'0' {
        zero_pad = true;
        i += 1;
    }
    let mut width = 0usize;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        width = width.saturating_mul(10).saturating_add((bytes[i] - b'0') as usize);
        i += 1;
    }
    let conv = if i < bytes.len() {
        bytes[i] as char
    } else {
        '\0'
    };
    if i < bytes.len() {
        i += 1;
    }
    (
        Spec {
            left_align,
            zero_pad,
            width,
            conv,
        },
        i,
    )
}

/// Render one argument with `core::fmt` according to `spec`,
/// applying width padding. Unknown conversions are rendered
/// literally as `%<conv>`.
fn render_arg(spec: &Spec, arg: Option<PrintfArg>) -> String {
    let raw: String = match (spec.conv, arg) {
        ('d' | 'i', Some(PrintfArg::Int(v))) => alloc::format!("{v}"),
        ('d' | 'i', Some(PrintfArg::UInt(v))) => alloc::format!("{v}"),
        ('u', Some(PrintfArg::UInt(v))) => alloc::format!("{v}"),
        ('u', Some(PrintfArg::Int(v))) => alloc::format!("{}", v as u64),
        ('x', Some(PrintfArg::UInt(v))) => alloc::format!("{v:x}"),
        ('x', Some(PrintfArg::Int(v))) => alloc::format!("{:x}", v as u64),
        ('X', Some(PrintfArg::UInt(v))) => alloc::format!("{v:X}"),
        ('X', Some(PrintfArg::Int(v))) => alloc::format!("{:X}", v as u64),
        ('s', Some(PrintfArg::Str(v))) => String::from(v),
        ('c', Some(PrintfArg::Char(v))) => {
            let mut s = String::new();
            s.push(v);
            s
        }
        ('c', Some(PrintfArg::Int(v))) => {
            let mut s = String::new();
            if let Some(c) = char::from_u32(v as u32) {
                s.push(c);
            }
            s
        }
        _ => {
            // Missing argument, wrong type, or unknown conversion:
            // emit the conversion literally so nothing is silently lost.
            let mut s = String::from("%");
            s.push(spec.conv);
            return s;
        }
    };
    pad(&raw, spec)
}

/// Pad `s` to `spec.width`. Zero-padding goes after a leading `-`
/// sign (like C); space padding goes on the outside.
fn pad(s: &str, spec: &Spec) -> String {
    let len = s.chars().count();
    if spec.width <= len {
        return String::from(s);
    }
    let pad_len = spec.width - len;
    let mut out = String::new();
    if spec.zero_pad && !spec.left_align {
        if let Some(rest) = s.strip_prefix('-') {
            out.push('-');
            for _ in 0..pad_len {
                out.push('0');
            }
            out.push_str(rest);
        } else {
            for _ in 0..pad_len {
                out.push('0');
            }
            out.push_str(s);
        }
    } else if spec.left_align {
        out.push_str(s);
        for _ in 0..pad_len {
            out.push(' ');
        }
    } else {
        for _ in 0..pad_len {
            out.push(' ');
        }
        out.push_str(s);
    }
    out
}

/// Write `printf`-formatted output to any `core::fmt::Write`.
/// `args` are consumed left to right; extra arguments are ignored
/// and a conversion with no argument left is written literally.
pub fn printf_to<W: core::fmt::Write>(
    w: &mut W,
    fmt: &str,
    args: &[PrintfArg],
) -> core::fmt::Result {
    let bytes = fmt.as_bytes();
    let mut i = 0;
    let mut arg_idx = 0;
    while i < bytes.len() {
        if bytes[i] != b'%' {
            // Copy the literal run up to the next `%`.
            let start = i;
            while i < bytes.len() && bytes[i] != b'%' {
                i += 1;
            }
            w.write_str(&fmt[start..i])?;
            continue;
        }
        // A `%` at the very end is a literal `%`.
        if i + 1 >= bytes.len() {
            w.write_str("%")?;
            i += 1;
            continue;
        }
        if bytes[i + 1] == b'%' {
            w.write_str("%")?;
            i += 2;
            continue;
        }
        let (spec, consumed) = parse_spec(&bytes[i + 1..]);
        if spec.conv == '\0' {
            // `%` followed by nothing parseable: literal.
            w.write_str("%")?;
            i += 1;
            continue;
        }
        let arg = args.get(arg_idx).copied();
        arg_idx += 1;
        w.write_str(&render_arg(&spec, arg))?;
        i += 1 + consumed;
    }
    Ok(())
}

/// Format like `printf` and return the resulting `String`.
pub fn format_printf(fmt: &str, args: &[PrintfArg]) -> String {
    let mut out = String::new();
    let _ = printf_to(&mut out, fmt, args);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_conversions() {
        assert_eq!(format_printf("hello", pargs![]), "hello");
        assert_eq!(format_printf("%d", pargs![42]), "42");
        assert_eq!(format_printf("%d", pargs![-7]), "-7");
        assert_eq!(format_printf("%i + %i = %i", pargs![1, 2, 3]), "1 + 2 = 3");
        assert_eq!(format_printf("%u", pargs![42u32]), "42");
        assert_eq!(format_printf("%x", pargs![255u32]), "ff");
        assert_eq!(format_printf("%X", pargs![255u32]), "FF");
        assert_eq!(format_printf("%x", pargs![0xdeadbeefu32]), "deadbeef");
        assert_eq!(format_printf("%s", pargs!["init"]), "init");
        assert_eq!(format_printf("%c", pargs!['A']), "A");
        assert_eq!(format_printf("100%%", pargs![]), "100%");
        assert_eq!(
            format_printf("pid=%d name=%s", pargs![7, "sh"]),
            "pid=7 name=sh"
        );
    }

    #[test]
    fn width_left_align_and_zero_pad() {
        assert_eq!(format_printf("[%5d]", pargs![42]), "[   42]");
        assert_eq!(format_printf("[%-5d]", pargs![42]), "[42   ]");
        assert_eq!(format_printf("[%05d]", pargs![42]), "[00042]");
        assert_eq!(format_printf("[%05d]", pargs![-42]), "[-0042]");
        assert_eq!(format_printf("[%08x]", pargs![255u32]), "[000000ff]");
        assert_eq!(format_printf("[%-8s]", pargs!["hi"]), "[hi      ]");
        assert_eq!(format_printf("[%8s]", pargs!["hi"]), "[      hi]");
        // Wider value than the width: no truncation.
        assert_eq!(format_printf("[%2d]", pargs![12345]), "[12345]");
    }

    #[test]
    fn unknown_conversion_and_missing_arg_are_literal() {
        assert_eq!(format_printf("%q", pargs![1]), "%q");
        assert_eq!(format_printf("%d %d", pargs![1]), "1 %d");
        assert_eq!(format_printf("trailing %", pargs![]), "trailing %");
        // Extra arguments are ignored.
        assert_eq!(format_printf("%d", pargs![1, 2, 3]), "1");
    }

    #[test]
    fn printf_to_writes_to_any_fmt_writer() {
        let mut buf = String::new();
        printf_to(&mut buf, "%s=%d", pargs!["n", 3]).unwrap();
        assert_eq!(buf, "n=3");
    }
}
