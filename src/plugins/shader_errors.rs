//! Shader compilation error surfacing.
//!
//! A safety net for invalid effects the editor's own validation doesn't yet
//! catch. `bevy_hanabi` bakes each effect to WGSL that `wgpu`/`naga` only
//! validate when the render pipeline is built — in the **render world**, where
//! a failure is otherwise just an `error!` log the user never sees while the
//! viewport silently goes blank.
//!
//! This plugin scans the render world's [`PipelineCache`] for pipelines stuck
//! in [`CachedPipelineState::Err`], pairs each with the shader asset(s) it was
//! built from, and hands the list to the main world through a shared buffer.
//! The main-world half files each error onto the [`ShaderErrors`] **component**
//! of the document it belongs to, matched by the shader [`AssetId`]s the
//! document's effect actually compiled (read from
//! [`bevy_hanabi::CompiledParticleEffect::get_configured_shaders`]). The UI
//! reads that component to show a per-tab warning icon and a banner in the
//! document's Shaders panel.
//!
//! The lists are rebuilt every frame from the live pipeline state, so an error
//! clears on its own once the offending edit is undone and the effect
//! recompiles cleanly.

use std::sync::{Arc, Mutex};

use bevy::{
    platform::collections::HashSet,
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        render_resource::{CachedPipelineState, PipelineCache, PipelineDescriptor},
    },
    shader::{ShaderCacheError, Shader},
};
use bevy_hanabi::CompiledParticleEffect;
use naga_oil::compose::{ComposerErrorInner, ErrSource};

use crate::document::DocumentSceneRoot;

/// A position within the compiled shader source the error points at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorLocation {
    /// 1-based line number in the *composed* shader source (after naga_oil
    /// inlines imports), so it need not match the pre-composition WGSL the
    /// Shaders panel displays.
    pub line: u32,
    /// 1-based column (byte offset within the line).
    pub column: u32,
    /// The offending source line itself, trimmed — the most directly useful
    /// part, since it pinpoints the code without a line-number cross-reference.
    pub snippet: String,
}

/// One failed pipeline, attributed to a document by its compiled shader ids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderCompileError {
    /// Path of the shader the failed pipeline was built from, e.g.
    /// `hanabi/demo~3_render_1234.wgsl`. `None` if the asset is gone or has no
    /// path (non-effect pipelines). Used for the phase label and panel banner,
    /// not for document matching.
    pub shader_path: Option<String>,
    /// Human-readable compiler error (the `PipelineCacheError` display chain).
    pub message: String,
    /// Source location the error points at, when the error carries a span we
    /// can resolve. `None` for errors without inline source (e.g. the rarer
    /// imported-module case, which needs the render world's private composer).
    pub location: Option<ErrorLocation>,
}

impl ShaderCompileError {
    /// The effect phase this shader belongs to, parsed from its path
    /// (`hanabi/{name}_{phase}_{hash}.wgsl`).
    pub fn phase(&self) -> Option<crate::document::ModifierGroup> {
        use crate::document::ModifierGroup::*;
        let path = self.shader_path.as_deref()?;
        for (suffix, group) in [("_init_", Init), ("_update_", Update), ("_render_", Render)] {
            if path.contains(suffix) {
                return Some(group);
            }
        }
        None
    }
}

/// Per-document component holding that document's failed shaders.
///
/// Empty when the effect compiles cleanly. Inserted on every document by
/// [`crate::app_commands::spawn_document`].
#[derive(Component, Default)]
pub struct ShaderErrors(pub Vec<ShaderCompileError>);

/// Raw error as captured in the render world, before main-world path
/// resolution.
#[derive(Clone)]
struct RawPipelineError {
    shaders: Vec<AssetId<Shader>>,
    message: String,
    location: Option<ErrorLocation>,
}

/// Shared buffer bridging the render-world scan and the main-world publisher.
///
/// The render-world scan writes it; the main-world publisher reads it. Inserted
/// (as a clone of the same `Arc`) into both apps.
#[derive(Resource, Clone)]
struct ShaderErrorChannel(Arc<Mutex<Vec<RawPipelineError>>>);

pub struct ShaderErrorPlugin;

impl Plugin for ShaderErrorPlugin {
    fn build(&self, app: &mut App) {
        let channel = ShaderErrorChannel(Arc::new(Mutex::new(Vec::new())));
        app.insert_resource(channel.clone())
            .add_systems(Update, publish_shader_errors);

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .insert_resource(channel)
                .add_systems(Render, collect_pipeline_errors.after(RenderSystems::Render));
        }
    }
}

/// Render world: snapshot every errored pipeline into the shared buffer.
fn collect_pipeline_errors(cache: Res<PipelineCache>, channel: Res<ShaderErrorChannel>) {
    let mut errors = Vec::new();
    for pipeline in cache.pipelines() {
        let CachedPipelineState::Err(err) = &pipeline.state else {
            continue;
        };
        errors.push(RawPipelineError {
            shaders: shader_ids(&pipeline.descriptor),
            message: error_chain(err),
            location: error_location(err),
        });
    }
    if let Ok(mut buf) = channel.0.lock() {
        *buf = errors;
    }
}

/// Main world: resolve shader ids and file errors onto their documents.
///
/// Files each error onto the owning document's [`ShaderErrors`] component,
/// matched by the shader [`AssetId`]s the document's effect compiled. Matching
/// by id (rather than by the `hanabi/{name}_…` path) is robust to hanabi's
/// source-keyed shader dedup: two documents with identical content share one
/// shader, so a failure in that shader is correctly reported on both.
fn publish_shader_errors(
    channel: Res<ShaderErrorChannel>,
    shaders: Res<Assets<Shader>>,
    compiled_effects: Query<(&ChildOf, &CompiledParticleEffect)>,
    scene_roots: Query<&ChildOf, With<DocumentSceneRoot>>,
    mut docs: Query<(Entity, &mut ShaderErrors)>,
) {
    let raw = match channel.0.lock() {
        Ok(buf) => buf.clone(),
        Err(_) => return,
    };

    // Resolve each captured error to a display path (for the phase label and
    // panel banner) while keeping the shader ids it was built from for
    // document matching. De-duplicate identical reports (one broken shader can
    // back several pipeline variants).
    let mut resolved: Vec<(ShaderCompileError, Vec<AssetId<Shader>>)> = Vec::new();
    for e in raw {
        let shader_path = e
            .shaders
            .iter()
            .find_map(|id| shaders.get(*id).map(|s| s.path.clone()));
        let entry = ShaderCompileError {
            shader_path,
            message: e.message,
            location: e.location,
        };
        if !resolved.iter().any(|(x, _)| *x == entry) {
            resolved.push((entry, e.shaders));
        }
    }

    for (doc_entity, mut errors) in &mut docs {
        let shader_ids = effect_shader_ids(doc_entity, &compiled_effects, &scene_roots);
        let matched: Vec<ShaderCompileError> = resolved
            .iter()
            .filter(|(_, ids)| ids.iter().any(|id| shader_ids.contains(id)))
            .map(|(e, _)| e.clone())
            .collect();
        if errors.0 != matched {
            errors.0 = matched;
        }
    }
}

/// The shader [`AssetId`]s hanabi compiled for `doc`'s effect.
///
/// Empty until the document's effect entity has been spawned and compiled. The
/// effect entity is a grandchild of the document (document → scene root →
/// [`bevy_hanabi::ParticleEffect`]).
fn effect_shader_ids(
    doc: Entity,
    compiled_effects: &Query<(&ChildOf, &CompiledParticleEffect)>,
    scene_roots: &Query<&ChildOf, With<DocumentSceneRoot>>,
) -> HashSet<AssetId<Shader>> {
    compiled_effects
        .iter()
        .filter(|(child_of, _)| {
            scene_roots
                .get(child_of.parent())
                .is_ok_and(|root| root.parent() == doc)
        })
        .filter_map(|(_, compiled)| compiled.get_configured_shaders())
        .flat_map(|s| [s.init.id(), s.update.id(), s.render.id()])
        .collect()
}

/// The shader asset(s) a pipeline descriptor was built from.
fn shader_ids(descriptor: &PipelineDescriptor) -> Vec<AssetId<Shader>> {
    match descriptor {
        PipelineDescriptor::RenderPipelineDescriptor(d) => {
            let mut ids = vec![d.vertex.shader.id()];
            if let Some(fragment) = &d.fragment {
                ids.push(fragment.shader.id());
            }
            ids
        }
        PipelineDescriptor::ComputePipelineDescriptor(d) => vec![d.shader.id()],
    }
}

/// Format a `PipelineCacheError` and its `source()` chain into one message.
fn error_chain(err: &(dyn std::error::Error + 'static)) -> String {
    let mut message = err.to_string();
    let mut source = err.source();
    while let Some(cause) = source {
        let text = cause.to_string();
        // `#[error(transparent)]` wrappers Display-delegate to their source, so
        // skip a cause already contained in what we've printed.
        if !message.contains(&text) {
            message.push_str("\n  caused by: ");
            message.push_str(&text);
        }
        source = cause.source();
    }
    message
}

/// Bit shift separating a span's module index from its in-module offset.
///
/// naga_oil encodes a span's owning module index in its high bits
/// (`module_index << SPAN_SHIFT`); the low bits are the in-module byte offset.
/// The shift is private in naga_oil (`compose::SPAN_SHIFT`).
const NAGA_OIL_SPAN_SHIFT: usize = 21;

/// Resolve a source location from a shader-compilation error.
///
/// Works when the error carries a span into inline (top-level) source. Returns
/// `None` for errors whose source lives in an imported module (resolvable only
/// via the render world's private composer) or that carry no span.
fn error_location(err: &ShaderCacheError) -> Option<ErrorLocation> {
    let ShaderCacheError::ProcessShaderError(ce) = err else {
        return None;
    };
    // Only the top-level shader being constructed carries its source inline;
    // imported-module errors reference a source we'd need the composer to read.
    let ErrSource::Constructing { source, offset, .. } = &ce.source else {
        return None;
    };
    let range = first_span(&ce.inner)?.to_range()?;
    // Reject spans owned by an imported module: their low bits index a
    // different source than the inline one we hold.
    if range.start >> NAGA_OIL_SPAN_SHIFT != 0 {
        return None;
    }
    let start = range.start.checked_sub(*offset)?;
    if start > source.len() {
        return None;
    }
    let prefix = &source[..start];
    let line = prefix.matches('\n').count() as u32 + 1;
    let line_start = prefix.rfind('\n').map(|p| p + 1).unwrap_or(0);
    let column = (start - line_start) as u32 + 1;
    let line_end = source[line_start..]
        .find('\n')
        .map(|p| line_start + p)
        .unwrap_or(source.len());
    let snippet = source[line_start..line_end].trim().to_string();
    Some(ErrorLocation {
        line,
        column,
        snippet,
    })
}

/// The first resolvable span from a composer error's inner variant.
fn first_span(inner: &ComposerErrorInner) -> Option<naga::Span> {
    match inner {
        ComposerErrorInner::ShaderValidationError(ws)
        | ComposerErrorInner::HeaderValidationError(ws) => ws.spans().next().map(|(s, _)| *s),
        ComposerErrorInner::WgslParseError(pe) => pe.labels().next().map(|(s, _)| s),
        _ => None,
    }
}
