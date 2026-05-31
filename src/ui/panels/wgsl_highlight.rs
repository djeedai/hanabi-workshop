//! Lightweight WGSL syntax highlighter for the Shaders panel.
//!
//! Produces an [`egui::text::LayoutJob`] from a WGSL source string,
//! coloring keywords, builtin types, attributes (`@group(...)`),
//! numeric literals, comments and strings. Designed for the
//! read-only Shaders panel: never fails, degrades gracefully on
//! malformed input, and emits one-character fallback tokens for
//! anything unrecognised.
//!
//! ## Why not naga?
//! naga's WGSL lexer is `pub(crate)`, the parser is all-or-nothing
//! (no highlighting on a syntax error), and its IR doesn't store
//! token-level spans — you'd have to re-tokenize inside each
//! expression span anyway. See discussion in commit history.
//!
//! ## Caching
//! [`highlight_cached`] stashes the produced `LayoutJob` in egui's
//! per-frame `data_temp` keyed by a hash of the source, so the
//! tokenizer only runs when the shader text actually changes
//! (a structural edit in hanabi recompiles the shader; until then
//! we just re-issue the cached job each frame).
//!
//! Wire-up: pass [`layouter`] to `TextEdit::layouter` — it adapts
//! the cached highlighter to egui's required closure signature.
//!
//! Token classes are coarse on purpose. A semantic highlighter
//! (user-fn vs builtin, struct field vs local) is a separate
//! feature; it would layer on top of this, not replace it.
//!
//! WGSL language reference used for keyword/type lists:
//! <https://www.w3.org/TR/WGSL/>

use std::sync::Arc;

use bevy_egui::egui::{
    self,
    text::{LayoutJob, TextFormat},
};

/// Highlight `src` using egui's current visuals, caching the result
/// in `ctx`'s per-frame memory keyed by `(source hash, dark/light)`.
///
/// The cache uses `data_temp` (cleared every frame by egui) rather
/// than `data` so we don't grow without bound across edits. Within
/// a frame the same shader source — drawn by multiple panels or
/// re-laid-out at a new wrap width — re-uses the same tokenization.
pub fn highlight_cached(ctx: &egui::Context, src: &str) -> LayoutJob {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    src.hash(&mut h);
    let dark = ctx.style().visuals.dark_mode;
    dark.hash(&mut h);
    let key = egui::Id::new(("wgsl-hl", h.finish()));

    if let Some(cached) = ctx.data(|d| d.get_temp::<Arc<LayoutJob>>(key)) {
        return (*cached).clone();
    }
    let palette = Palette::for_mode(dark);
    let job = highlight(src, &palette);
    ctx.data_mut(|d| d.insert_temp(key, Arc::new(job.clone())));
    job
}

/// `TextEdit::layouter`-compatible closure adapter.
///
/// Usage:
/// ```ignore
/// let mut layouter = wgsl_highlight::layouter();
/// ui.add(egui::TextEdit::multiline(&mut src).layouter(&mut layouter));
/// ```
pub fn layouter() -> impl FnMut(&egui::Ui, &dyn egui::TextBuffer, f32) -> Arc<egui::Galley> {
    move |ui, buf, wrap_width| {
        let mut job = highlight_cached(ui.ctx(), buf.as_str());
        job.wrap.max_width = wrap_width;
        ui.fonts_mut(|f| f.layout_job(job))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tok {
    Whitespace,
    LineComment,
    BlockComment,
    Keyword,
    Type,
    Builtin,
    Number,
    StringLit,
    Attribute,
    Ident,
    Punct,
}

struct Palette {
    text: egui::Color32,
    comment: egui::Color32,
    keyword: egui::Color32,
    ty: egui::Color32,
    builtin: egui::Color32,
    number: egui::Color32,
    string: egui::Color32,
    attribute: egui::Color32,
    punct: egui::Color32,
}

impl Palette {
    fn for_mode(dark: bool) -> Self {
        if dark {
            Self {
                text: egui::Color32::from_rgb(0xE0, 0xE0, 0xE0),
                comment: egui::Color32::from_rgb(0x6A, 0x99, 0x55),
                keyword: egui::Color32::from_rgb(0x56, 0x9C, 0xD6),
                ty: egui::Color32::from_rgb(0x4E, 0xC9, 0xB0),
                builtin: egui::Color32::from_rgb(0xDC, 0xDC, 0xAA),
                number: egui::Color32::from_rgb(0xB5, 0xCE, 0xA8),
                string: egui::Color32::from_rgb(0xCE, 0x91, 0x78),
                attribute: egui::Color32::from_rgb(0xC5, 0x86, 0xC0),
                punct: egui::Color32::from_rgb(0xD4, 0xD4, 0xD4),
            }
        } else {
            Self {
                text: egui::Color32::from_rgb(0x1F, 0x1F, 0x1F),
                comment: egui::Color32::from_rgb(0x00, 0x80, 0x00),
                keyword: egui::Color32::from_rgb(0x00, 0x00, 0xFF),
                ty: egui::Color32::from_rgb(0x26, 0x7F, 0x99),
                builtin: egui::Color32::from_rgb(0x79, 0x5E, 0x26),
                number: egui::Color32::from_rgb(0x09, 0x86, 0x58),
                string: egui::Color32::from_rgb(0xA3, 0x15, 0x15),
                attribute: egui::Color32::from_rgb(0xAF, 0x00, 0xDB),
                punct: egui::Color32::from_rgb(0x3B, 0x3B, 0x3B),
            }
        }
    }

    fn color(&self, t: Tok) -> egui::Color32 {
        match t {
            Tok::Whitespace | Tok::Punct => self.punct,
            Tok::LineComment | Tok::BlockComment => self.comment,
            Tok::Keyword => self.keyword,
            Tok::Type => self.ty,
            Tok::Builtin => self.builtin,
            Tok::Number => self.number,
            Tok::StringLit => self.string,
            Tok::Attribute => self.attribute,
            Tok::Ident => self.text,
        }
    }
}

/// WGSL reserved keywords (control flow, declarations, qualifiers).
/// Source: WGSL spec §2.4 "Keywords" + §5 "Declarations".
const KEYWORDS: &[&str] = &[
    "alias", "break", "case", "const", "const_assert", "continue", "continuing",
    "default", "diagnostic", "discard", "else", "enable", "false", "fn", "for",
    "if", "let", "loop", "override", "requires", "return", "struct", "switch",
    "true", "var", "while",
    // Address spaces / access modes — treated as keywords by the lexer.
    "function", "private", "workgroup", "uniform", "storage", "push_constant",
    "read", "write", "read_write",
];

/// Predeclared scalar / vector / matrix / texture / sampler type names.
/// Matrices are matched by prefix (`mat2x2`..`mat4x4`); textures by
/// prefix (`texture_`). Listed individually here for the exact-match
/// fast path.
const TYPES: &[&str] = &[
    "bool", "i32", "u32", "f32", "f16", "i64", "u64",
    "vec2", "vec3", "vec4",
    "vec2f", "vec3f", "vec4f", "vec2i", "vec3i", "vec4i", "vec2u", "vec3u", "vec4u",
    "vec2h", "vec3h", "vec4h",
    "mat2x2", "mat2x3", "mat2x4",
    "mat3x2", "mat3x3", "mat3x4",
    "mat4x2", "mat4x3", "mat4x4",
    "mat2x2f", "mat3x3f", "mat4x4f",
    "array", "atomic", "ptr", "sampler", "sampler_comparison",
    "void",
];

/// A small selection of WGSL builtin functions — enough to make
/// hanabi-generated shaders read well. Not exhaustive on purpose;
/// unrecognised idents render as plain text, which is fine.
const BUILTINS: &[&str] = &[
    "abs", "acos", "all", "any", "asin", "atan", "atan2", "ceil", "clamp",
    "cos", "cosh", "cross", "degrees", "determinant", "distance", "dot",
    "exp", "exp2", "faceForward", "floor", "fma", "fract", "frexp",
    "inverseSqrt", "ldexp", "length", "log", "log2", "max", "min", "mix",
    "modf", "normalize", "pow", "radians", "reflect", "refract", "round",
    "saturate", "sign", "sin", "sinh", "smoothstep", "sqrt", "step",
    "tan", "tanh", "transpose", "trunc",
    // Texture / atomics / derivatives.
    "textureSample", "textureSampleLevel", "textureSampleBias",
    "textureSampleGrad", "textureSampleCompare", "textureLoad",
    "textureStore", "textureDimensions", "textureNumLayers",
    "textureNumLevels", "textureNumSamples", "textureGather",
    "atomicLoad", "atomicStore", "atomicAdd", "atomicSub", "atomicMax",
    "atomicMin", "atomicAnd", "atomicOr", "atomicXor", "atomicExchange",
    "atomicCompareExchangeWeak",
    "dpdx", "dpdy", "fwidth", "select", "bitcast",
    "workgroupBarrier", "storageBarrier", "textureBarrier",
];

fn classify_ident(s: &str) -> Tok {
    // Linear scans on tiny static slices are fine — these lists are
    // <100 items and identifiers are short. Avoids the "must remain
    // sorted" footgun of `binary_search`.
    if KEYWORDS.contains(&s) {
        return Tok::Keyword;
    }
    if TYPES.contains(&s) {
        return Tok::Type;
    }
    // Matrix/texture families: prefix match.
    if s.starts_with("texture_") || (s.starts_with("mat") && matches_mat(s)) {
        return Tok::Type;
    }
    if BUILTINS.contains(&s) {
        return Tok::Builtin;
    }
    Tok::Ident
}

fn matches_mat(s: &str) -> bool {
    // Accept `mat{2,3,4}x{2,3,4}` optionally suffixed with `f`/`h`.
    let bytes = s.as_bytes();
    if bytes.len() < 6 {
        return false;
    }
    let r = bytes[3];
    let x = bytes[4];
    let c = bytes[5];
    matches!(r, b'2' | b'3' | b'4')
        && x == b'x'
        && matches!(c, b'2' | b'3' | b'4')
        && (bytes.len() == 6
            || (bytes.len() == 7 && matches!(bytes[6], b'f' | b'h' | b'i' | b'u')))
}

/// Tokenize `src` into a `LayoutJob` colored using `pal`.
///
/// Algorithm: single forward pass over chars, dispatching on the
/// first char of each token. Block comments are handled with a
/// nesting counter (WGSL spec §2.3.2 — `/* … */` nests). Anything
/// that doesn't start a recognized token is emitted as a single
/// punctuation char so the loop always makes progress.
fn highlight(src: &str, pal: &Palette) -> LayoutJob {
    let mono = egui::FontId::monospace(13.0);
    let mk = |c: egui::Color32| TextFormat {
        font_id: mono.clone(),
        color: c,
        ..Default::default()
    };

    let mut job = LayoutJob::default();
    let bytes = src.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];

        // Whitespace run.
        if b.is_ascii_whitespace() {
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            job.append(&src[start..i], 0.0, mk(pal.color(Tok::Whitespace)));
            continue;
        }

        // Line comment `// ...`
        if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            let start = i;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            job.append(&src[start..i], 0.0, mk(pal.color(Tok::LineComment)));
            continue;
        }

        // Block comment `/* ... */` (nests).
        if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            let start = i;
            i += 2;
            let mut depth = 1usize;
            while i + 1 < bytes.len() && depth > 0 {
                if bytes[i] == b'/' && bytes[i + 1] == b'*' {
                    depth += 1;
                    i += 2;
                } else if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if depth > 0 {
                // Unterminated — color rest of source as comment.
                i = bytes.len();
            }
            job.append(&src[start..i], 0.0, mk(pal.color(Tok::BlockComment)));
            continue;
        }

        // String literal — rare in WGSL but appears in `enable "..."`
        // and `diagnostic(...)` rule names; quoted with `"`.
        if b == b'"' {
            let start = i;
            i += 1;
            while i < bytes.len() && bytes[i] != b'"' {
                if bytes[i] == b'\\' && i + 1 < bytes.len() {
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if i < bytes.len() {
                i += 1; // closing quote
            }
            job.append(&src[start..i], 0.0, mk(pal.color(Tok::StringLit)));
            continue;
        }

        // Attribute `@ident`.
        if b == b'@' {
            let start = i;
            i += 1;
            while i < bytes.len() && is_ident_cont(bytes[i]) {
                i += 1;
            }
            job.append(&src[start..i], 0.0, mk(pal.color(Tok::Attribute)));
            continue;
        }

        // Numeric literal: digit, or `.digit` (decimal fraction), or
        // a leading `-`/`+` would be lexed as a punct then number;
        // the spec treats sign as a unary op so we follow suit.
        if b.is_ascii_digit()
            || (b == b'.' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit())
        {
            let start = i;
            // hex prefix
            if b == b'0' && i + 1 < bytes.len() && (bytes[i + 1] == b'x' || bytes[i + 1] == b'X') {
                i += 2;
                while i < bytes.len() && (bytes[i].is_ascii_hexdigit() || bytes[i] == b'_') {
                    i += 1;
                }
            } else {
                while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'_') {
                    i += 1;
                }
                if i < bytes.len() && bytes[i] == b'.' {
                    i += 1;
                    while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'_') {
                        i += 1;
                    }
                }
                // Exponent.
                if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
                    i += 1;
                    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
                        i += 1;
                    }
                    while i < bytes.len() && bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            // Type suffix: f, h, i, u (e.g. `1.0f`, `42u`).
            if i < bytes.len() && matches!(bytes[i], b'f' | b'h' | b'i' | b'u') {
                i += 1;
            }
            job.append(&src[start..i], 0.0, mk(pal.color(Tok::Number)));
            continue;
        }

        // Identifier / keyword / type / builtin.
        if is_ident_start(b) {
            let start = i;
            while i < bytes.len() && is_ident_cont(bytes[i]) {
                i += 1;
            }
            let ident = &src[start..i];
            job.append(ident, 0.0, mk(pal.color(classify_ident(ident))));
            continue;
        }

        // Fallback: single punctuation byte (works because all
        // remaining unmatched ASCII operators are 1 byte each, and
        // non-ASCII bytes are illegal in WGSL outside string/comment
        // — color them as punct and step by one *char* to stay on a
        // codepoint boundary).
        let ch_len = utf8_char_len(b);
        let end = (i + ch_len).min(bytes.len());
        job.append(&src[i..end], 0.0, mk(pal.color(Tok::Punct)));
        i = end;
    }
    job
}

fn is_ident_start(b: u8) -> bool {
    b == b'_' || b.is_ascii_alphabetic()
}

fn is_ident_cont(b: u8) -> bool {
    b == b'_' || b.is_ascii_alphanumeric()
}

fn utf8_char_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b < 0xC0 {
        1 // continuation byte standing alone — defensive
    } else if b < 0xE0 {
        2
    } else if b < 0xF0 {
        3
    } else {
        4
    }
}
