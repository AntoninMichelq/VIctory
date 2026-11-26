//! Serialised pipeline recipes (`early-recipes.bin`): each pipeline a run
//! drew or dispatched with, its shader stages and container hashes, and its
//! [`GraphicsState`], so `victory prewarm` can recreate and store every host
//! pipeline in `host-pipelines.bin` without starting the title.

use std::ffi::CString;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::PCSTR;

use super::Shader;
use super::host::GraphicsState;

const MAGIC: &[u8; 4] = b"VPR1";

/// One pipeline recipe saved beside `early-pipelines.txt`.
pub(crate) struct Recipe {
    /// The `Pending::key` computed when the pipeline was recorded.
    pub(crate) key: String,
    /// `(stage, 16-char blake3 hex hash)` in the title's order.
    pub(crate) shaders: Vec<(String, String)>,
    /// Owned graphics state (`None` for compute pipelines).
    pub(crate) graphics: Option<GraphicsState>,
}

impl Recipe {
    pub(crate) fn from_pending(
        key: &str,
        shaders: &[std::sync::Arc<Shader>],
        graphics: Option<&GraphicsState>,
    ) -> Recipe {
        Recipe {
            key: key.to_owned(),
            shaders: shaders
                .iter()
                .map(|s| (s.stage.to_owned(), s.hash.clone()))
                .collect(),
            graphics: graphics.map(clone_graphics),
        }
    }

    /// Computes the `Pending::key` for this recipe under the current settings
    /// (e.g. `VICTORY_DEPTH_BOUNDS`).
    pub(crate) fn current_key(&self) -> String {
        let mut h = blake3::Hasher::new();
        for (stage, hash) in &self.shaders {
            h.update(stage.as_bytes());
            h.update(hash.as_bytes());
        }
        h.update(
            self.graphics
                .as_ref()
                .map(|g| g.describe())
                .unwrap_or_default()
                .as_bytes(),
        );
        h.finalize().to_hex()[..32].to_owned()
    }
}

pub(crate) fn clone_graphics(g: &GraphicsState) -> GraphicsState {
    let names = g.names.clone();
    let inputs = g
        .inputs
        .iter()
        .zip(&names)
        .map(|(e, name)| D3D12_INPUT_ELEMENT_DESC {
            SemanticName: PCSTR(name.as_ptr().cast()),
            ..*e
        })
        .collect();
    GraphicsState {
        blend: g.blend,
        sample_mask: g.sample_mask,
        rasterizer: g.rasterizer,
        depth: g.depth,
        inputs,
        names,
        strip_cut: g.strip_cut,
        topology: g.topology,
        targets: g.targets,
        depth_format: g.depth_format,
        samples: g.samples,
        title_depth_bounds: g.title_depth_bounds,
        depth_bounds: g.depth_bounds,
        unwritten_masked: g.unwritten_masked,
    }
}

pub(crate) fn encode_all(recipes: &[&Recipe]) -> Vec<u8> {
    let mut out = Vec::with_capacity(recipes.len() * 512);
    out.extend_from_slice(MAGIC);
    put_u32(&mut out, recipes.len() as u32);
    for r in recipes {
        put_str(&mut out, &r.key);
        put_u32(&mut out, r.shaders.len() as u32);
        for (stage, hash) in &r.shaders {
            put_str(&mut out, stage);
            put_str(&mut out, hash);
        }
        match &r.graphics {
            Some(g) => {
                out.push(1);
                encode_graphics(&mut out, g);
            }
            None => out.push(0),
        }
    }
    out
}

pub(crate) fn decode_all(bytes: &[u8]) -> Option<Vec<Recipe>> {
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    let count = r.u32()? as usize;
    let mut out = Vec::with_capacity(count.min(16_384));
    for _ in 0..count {
        let key = r.string()?;
        let n_shaders = r.u32()? as usize;
        if n_shaders > 16 {
            return None;
        }
        let mut shaders = Vec::with_capacity(n_shaders);
        for _ in 0..n_shaders {
            let stage = r.string()?;
            let hash = r.string()?;
            shaders.push((stage, hash));
        }
        let graphics = match r.u8()? {
            0 => None,
            1 => Some(decode_graphics(&mut r)?),
            _ => return None,
        };
        out.push(Recipe {
            key,
            shaders,
            graphics,
        });
    }
    r.0.is_empty().then_some(out)
}

fn encode_graphics(out: &mut Vec<u8>, g: &GraphicsState) {
    put_i32(out, g.blend.AlphaToCoverageEnable.0);
    put_i32(out, g.blend.IndependentBlendEnable.0);
    for rt in &g.blend.RenderTarget {
        put_i32(out, rt.BlendEnable.0);
        put_i32(out, rt.LogicOpEnable.0);
        put_i32(out, rt.SrcBlend.0);
        put_i32(out, rt.DestBlend.0);
        put_i32(out, rt.BlendOp.0);
        put_i32(out, rt.SrcBlendAlpha.0);
        put_i32(out, rt.DestBlendAlpha.0);
        put_i32(out, rt.BlendOpAlpha.0);
        put_i32(out, rt.LogicOp.0);
        out.push(rt.RenderTargetWriteMask);
    }
    put_u32(out, g.sample_mask);
    let ras = &g.rasterizer;
    put_i32(out, ras.FillMode.0);
    put_i32(out, ras.CullMode.0);
    put_i32(out, ras.FrontCounterClockwise.0);
    put_i32(out, ras.DepthBias);
    put_u32(out, ras.DepthBiasClamp.to_bits());
    put_u32(out, ras.SlopeScaledDepthBias.to_bits());
    put_i32(out, ras.DepthClipEnable.0);
    put_i32(out, ras.MultisampleEnable.0);
    put_i32(out, ras.AntialiasedLineEnable.0);
    put_u32(out, ras.ForcedSampleCount);
    put_i32(out, ras.ConservativeRaster.0);

    let d = &g.depth;
    put_i32(out, d.DepthEnable.0);
    put_i32(out, d.DepthWriteMask.0);
    put_i32(out, d.DepthFunc.0);
    put_i32(out, d.StencilEnable.0);
    out.push(d.StencilReadMask);
    out.push(d.StencilWriteMask);
    for face in [&d.FrontFace, &d.BackFace] {
        put_i32(out, face.StencilFailOp.0);
        put_i32(out, face.StencilDepthFailOp.0);
        put_i32(out, face.StencilPassOp.0);
        put_i32(out, face.StencilFunc.0);
    }

    put_u32(out, g.inputs.len() as u32);
    for (e, name) in g.inputs.iter().zip(&g.names) {
        put_bytes(out, name.as_bytes());
        put_u32(out, e.SemanticIndex);
        put_i32(out, e.Format.0);
        put_u32(out, e.InputSlot);
        put_u32(out, e.AlignedByteOffset);
        put_i32(out, e.InputSlotClass.0);
        put_u32(out, e.InstanceDataStepRate);
    }

    put_i32(out, g.strip_cut.0);
    put_i32(out, g.topology.0);
    put_u32(out, g.targets.NumRenderTargets);
    for fmt in &g.targets.RTFormats {
        put_i32(out, fmt.0);
    }
    put_i32(out, g.depth_format.0);
    put_u32(out, g.samples.Count);
    put_u32(out, g.samples.Quality);
    let flags = u8::from(g.title_depth_bounds)
        | (u8::from(g.depth_bounds) << 1)
        | (u8::from(g.unwritten_masked) << 2);
    out.push(flags);
}

fn decode_graphics(r: &mut Reader<'_>) -> Option<GraphicsState> {
    use windows::core::BOOL;
    let alpha_to_cov = BOOL(r.i32()?);
    let indep_blend = BOOL(r.i32()?);
    let mut rts = [D3D12_RENDER_TARGET_BLEND_DESC::default(); 8];
    for rt in &mut rts {
        *rt = D3D12_RENDER_TARGET_BLEND_DESC {
            BlendEnable: BOOL(r.i32()?),
            LogicOpEnable: BOOL(r.i32()?),
            SrcBlend: D3D12_BLEND(r.i32()?),
            DestBlend: D3D12_BLEND(r.i32()?),
            BlendOp: D3D12_BLEND_OP(r.i32()?),
            SrcBlendAlpha: D3D12_BLEND(r.i32()?),
            DestBlendAlpha: D3D12_BLEND(r.i32()?),
            BlendOpAlpha: D3D12_BLEND_OP(r.i32()?),
            LogicOp: D3D12_LOGIC_OP(r.i32()?),
            RenderTargetWriteMask: r.u8()?,
        };
    }
    let blend = D3D12_BLEND_DESC {
        AlphaToCoverageEnable: alpha_to_cov,
        IndependentBlendEnable: indep_blend,
        RenderTarget: rts,
    };
    let sample_mask = r.u32()?;
    let rasterizer = D3D12_RASTERIZER_DESC {
        FillMode: D3D12_FILL_MODE(r.i32()?),
        CullMode: D3D12_CULL_MODE(r.i32()?),
        FrontCounterClockwise: BOOL(r.i32()?),
        DepthBias: r.i32()?,
        DepthBiasClamp: f32::from_bits(r.u32()?),
        SlopeScaledDepthBias: f32::from_bits(r.u32()?),
        DepthClipEnable: BOOL(r.i32()?),
        MultisampleEnable: BOOL(r.i32()?),
        AntialiasedLineEnable: BOOL(r.i32()?),
        ForcedSampleCount: r.u32()?,
        ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE(r.i32()?),
    };
    let depth_enable = BOOL(r.i32()?);
    let depth_write = D3D12_DEPTH_WRITE_MASK(r.i32()?);
    let depth_func = D3D12_COMPARISON_FUNC(r.i32()?);
    let stencil_enable = BOOL(r.i32()?);
    let stencil_read = r.u8()?;
    let stencil_write = r.u8()?;
    let mut face = || -> Option<D3D12_DEPTH_STENCILOP_DESC> {
        Some(D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP(r.i32()?),
            StencilDepthFailOp: D3D12_STENCIL_OP(r.i32()?),
            StencilPassOp: D3D12_STENCIL_OP(r.i32()?),
            StencilFunc: D3D12_COMPARISON_FUNC(r.i32()?),
        })
    };
    let front_face = face()?;
    let back_face = face()?;
    let depth = D3D12_DEPTH_STENCIL_DESC {
        DepthEnable: depth_enable,
        DepthWriteMask: depth_write,
        DepthFunc: depth_func,
        StencilEnable: stencil_enable,
        StencilReadMask: stencil_read,
        StencilWriteMask: stencil_write,
        FrontFace: front_face,
        BackFace: back_face,
    };

    let n_inputs = r.u32()? as usize;
    if n_inputs > 64 {
        return None;
    }
    let mut names = Vec::with_capacity(n_inputs);
    let mut raw_inputs = Vec::with_capacity(n_inputs);
    for _ in 0..n_inputs {
        let name = CString::new(r.bytes()?).ok()?;
        let semantic_index = r.u32()?;
        let format = DXGI_FORMAT(r.i32()?);
        let input_slot = r.u32()?;
        let aligned_byte_offset = r.u32()?;
        let input_slot_class = D3D12_INPUT_CLASSIFICATION(r.i32()?);
        let instance_data_step_rate = r.u32()?;
        names.push(name);
        raw_inputs.push((
            semantic_index,
            format,
            input_slot,
            aligned_byte_offset,
            input_slot_class,
            instance_data_step_rate,
        ));
    }
    let inputs = raw_inputs
        .into_iter()
        .zip(&names)
        .map(
            |(
                (
                    semantic_index,
                    format,
                    input_slot,
                    aligned_byte_offset,
                    input_slot_class,
                    instance_data_step_rate,
                ),
                name,
            )| D3D12_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(name.as_ptr().cast()),
                SemanticIndex: semantic_index,
                Format: format,
                InputSlot: input_slot,
                AlignedByteOffset: aligned_byte_offset,
                InputSlotClass: input_slot_class,
                InstanceDataStepRate: instance_data_step_rate,
            },
        )
        .collect();

    let strip_cut = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE(r.i32()?);
    let topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE(r.i32()?);
    let num_rt = r.u32()?;
    let mut rt_formats = [DXGI_FORMAT::default(); 8];
    for fmt in &mut rt_formats {
        *fmt = DXGI_FORMAT(r.i32()?);
    }
    let targets = D3D12_RT_FORMAT_ARRAY {
        RTFormats: rt_formats,
        NumRenderTargets: num_rt,
    };
    let depth_format = DXGI_FORMAT(r.i32()?);
    let samples = DXGI_SAMPLE_DESC {
        Count: r.u32()?,
        Quality: r.u32()?,
    };
    let flags = r.u8()?;
    let title_depth_bounds = (flags & 1) != 0;
    let depth_bounds = title_depth_bounds && super::host::depth_bounds_on();

    Some(GraphicsState {
        blend,
        sample_mask,
        rasterizer,
        depth,
        inputs,
        names,
        strip_cut,
        topology,
        targets,
        depth_format,
        samples,
        title_depth_bounds,
        depth_bounds,
        unwritten_masked: (flags & 4) != 0,
    })
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_i32(out: &mut Vec<u8>, v: i32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_bytes(out: &mut Vec<u8>, b: &[u8]) {
    put_u32(out, b.len() as u32);
    out.extend_from_slice(b);
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    put_bytes(out, s.as_bytes());
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let (head, rest) = self.0.split_at_checked(n)?;
        self.0 = rest;
        Some(head)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn i32(&mut self) -> Option<i32> {
        Some(i32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn bytes(&mut self) -> Option<&'a [u8]> {
        let n = self.u32()? as usize;
        self.take(n)
    }
    fn string(&mut self) -> Option<String> {
        std::str::from_utf8(self.bytes()?).ok().map(str::to_owned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipes_round_trip_with_identical_describe_and_key() {
        let mut g = GraphicsState::default();
        g.blend.IndependentBlendEnable = true.into();
        g.blend.RenderTarget[0].BlendEnable = true.into();
        g.blend.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;
        g.blend.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
        g.sample_mask = 0x55aa_55aa;
        g.rasterizer.CullMode = D3D12_CULL_MODE_NONE;
        g.rasterizer.DepthBias = -2;
        g.rasterizer.SlopeScaledDepthBias = 1.5;
        g.depth.DepthFunc = D3D12_COMPARISON_FUNC_GREATER_EQUAL;
        g.depth.StencilEnable = true.into();
        g.depth.FrontFace.StencilPassOp = D3D12_STENCIL_OP_REPLACE;
        g.names = vec![
            CString::new("POSITION").unwrap(),
            CString::new("TEXCOORD").unwrap(),
        ];
        g.inputs = vec![
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(g.names[0].as_ptr().cast()),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R32G32B32_FLOAT,
                InputSlot: 0,
                AlignedByteOffset: 0,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            },
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(g.names[1].as_ptr().cast()),
                SemanticIndex: 1,
                Format: DXGI_FORMAT_R16G16_FLOAT,
                InputSlot: 1,
                AlignedByteOffset: 12,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA,
                InstanceDataStepRate: 1,
            },
        ];
        g.strip_cut = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF;
        g.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        g.targets.NumRenderTargets = 2;
        g.targets.RTFormats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT;
        g.targets.RTFormats[1] = DXGI_FORMAT_R8G8B8A8_UNORM;
        g.depth_format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
        g.samples = DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        };
        let orig_desc = g.describe();

        let gfx = Recipe {
            key: String::new(),
            shaders: vec![
                ("vertex".into(), "0123456789abcdef".into()),
                ("pixel".into(), "fedcba9876543210".into()),
            ],
            graphics: Some(g),
        };
        let gfx = Recipe {
            key: gfx.current_key(),
            ..gfx
        };
        let comp = Recipe {
            key: String::new(),
            shaders: vec![("compute".into(), "39f0e9aeb156a61c".into())],
            graphics: None,
        };
        let comp = Recipe {
            key: comp.current_key(),
            ..comp
        };

        let bytes = encode_all(&[&gfx, &comp]);
        let decoded = decode_all(&bytes).expect("decode");
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].key, gfx.key);
        assert_eq!(decoded[0].shaders, gfx.shaders);
        assert_eq!(decoded[0].graphics.as_ref().unwrap().describe(), orig_desc);
        assert_eq!(decoded[0].current_key(), gfx.key);
        assert_eq!(decoded[1].key, comp.key);
        assert!(decoded[1].graphics.is_none());
        assert_eq!(decoded[1].current_key(), comp.key);

        // Truncated or trailing bytes are rejected.
        assert!(decode_all(&bytes[..bytes.len() - 1]).is_none());
        assert!(decode_all(&[bytes.as_slice(), &[0]].concat()).is_none());
    }
}
