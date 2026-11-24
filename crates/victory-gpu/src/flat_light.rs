//! `VICTORY_FLAT_LIGHT=<width>x<height>:<format>[:<n>[:<strength>[:<sky>[:<reversed>]]]]`:
//! a stand-in for lighting the title cannot do yet. At each present, the
//! `n`-th texture of that size and DXGI format the frame rendered to or
//! wrote (its G-buffer colours, say; first-use order, default 0) is added
//! to the title's image, `strength` times (default 0.8), darkened towards
//! the bottom of the screen by `sky` (0 flat, 1 half as bright at the
//! bottom; default 0.5). Surfaces the title's lighting leaves black then
//! show their shapes and colours, under the lights the image already has.
//! With the frame's first depth buffer of that size, surfaces are also
//! shaded by a light from above and to the right, with normals taken from
//! the depth (as reversed depth unless `<reversed>` is 0), and outlined
//! where the depth jumps. An integer texture (R8/R16/R32_UINT, a material
//! ID target say) is coloured from a palette, one muted colour per ID, ID
//! 0 black. Frames without such a texture present as usual.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::s;

use crate::host::{Host, HostList, transition};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=9, b0), DescriptorTable(SRV(t0, numDescriptors=4), UAV(u0))"
cbuffer Constants : register(b0) {
    uint W;
    uint H;
    uint AW;
    uint AH;
    float K;
    float Sky;
    uint HasDepth;
    uint Reversed;
    uint Palette;
};
Texture2D<float4> Scene : register(t0);
Texture2D<float4> Albedo : register(t1);
Texture2D<float> Depth : register(t2);
Texture2D<uint> Ids : register(t3);
RWTexture2D<unorm float4> Out : register(u0);

// Distance from the eye, up to a scale: depth is 1/z (reversed) or 1 - 1/z.
float Distance(int2 p) {
    p = clamp(p, int2(0, 0), int2(W - 1, H - 1));
    float d = Depth.Load(int3(p, 0));
    d = Reversed != 0 ? d : 1.0 - d;
    return 1.0 / max(d, 1e-6);
}

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= W || id.y >= H) {
        return;
    }
    int2 p = int2(id.xy);
    float3 lit = Scene.Load(int3(p, 0)).rgb;
    uint2 at = uint2(id.x * AW / W, id.y * AH / H);
    float3 colour;
    if (Palette != 0) {
        // A material ID: a muted colour of its own (0, the sky or nothing,
        // stays black).
        uint m = Ids.Load(int3(at, 0));
        uint hash = m * 2654435761u;
        float3 hue = float3(hash & 255, (hash >> 8) & 255, (hash >> 16) & 255) / 255.0;
        colour = m == 0 ? float3(0, 0, 0) : 0.3 + 0.5 * hue;
    } else {
        colour = Albedo.Load(int3(at, 0)).rgb;
    }
    float light = 1.0 - 0.5 * Sky * (float)id.y / (float)H;
    if (HasDepth != 0) {
        float z = Distance(p);
        float zl = Distance(p - int2(1, 0));
        float zr = Distance(p + int2(1, 0));
        float zu = Distance(p - int2(0, 1));
        float zd = Distance(p + int2(0, 1));
        // The smaller one-sided step: no false slope across an edge.
        float dx = abs(zr - z) < abs(z - zl) ? zr - z : z - zl;
        float dy = abs(zd - z) < abs(z - zu) ? zd - z : z - zu;
        // A pixel's width at that distance (a 90 degree view, roughly).
        float pixel = 2.0 * z / (float)H;
        float3 n = normalize(float3(-dx / pixel, dy / pixel, 1.0));
        float3 sun = normalize(float3(0.35, 0.8, 0.5));
        float lambert = saturate(dot(n, sun));
        float shade = 0.35 + 0.65 * lambert;
        // Outlines where the distance jumps.
        float jump = max(abs(zr - zl), abs(zd - zu)) / max(z, 1e-6);
        shade *= 1.0 - 0.5 * saturate(jump * 10.0 - 0.5);
        light *= shade;
    }
    Out[id.xy] = float4(saturate(lit + colour * K * light), 1.0);
}
"#;

/// Strength, sky and reversed depth from the variable, past the target's spec.
pub(crate) fn settings() -> (f32, f32, bool) {
    static SETTINGS: std::sync::OnceLock<(f32, f32, bool)> = std::sync::OnceLock::new();
    *SETTINGS.get_or_init(|| {
        let v = std::env::var("VICTORY_FLAT_LIGHT").unwrap_or_default();
        let mut parts = v.split(':').skip(3);
        let mut next = |default: f32| {
            parts
                .next()
                .and_then(|p| p.trim().parse().ok())
                .unwrap_or(default)
        };
        let strength = next(0.8);
        let sky = next(0.5);
        let reversed = next(1.0) != 0.0;
        (strength, sky, reversed)
    })
}

struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    heap: ID3D12DescriptorHeap,
    step: u32,
    list: HostList,
    /// The output, its width and height (left in UNORDERED_ACCESS).
    out: Option<(ID3D12Resource, u32, u32)>,
}

static PASS: std::sync::Mutex<Option<Result<Pass, ()>>> = std::sync::Mutex::new(None);

fn compile(host: &Host) -> windows::core::Result<Pass> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("flat_light"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("flat light: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes)?;
        let pso = host
            .device
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
        let heap: ID3D12DescriptorHeap =
            host.device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 5,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })?;
        let step = host
            .device
            .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        let list = host.new_list()?;
        list.close();
        tracing::info!("flat light: stand-in lighting on");
        Ok(Pass {
            root,
            pso,
            heap,
            step,
            list,
            out: None,
        })
    }
}

/// A format a texture SRV can read the texture's bytes as.
fn readable(format: DXGI_FORMAT) -> DXGI_FORMAT {
    match format {
        DXGI_FORMAT_R8G8B8A8_TYPELESS => DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_B8G8R8A8_TYPELESS => DXGI_FORMAT_B8G8R8A8_UNORM,
        DXGI_FORMAT_B8G8R8X8_TYPELESS => DXGI_FORMAT_B8G8R8X8_UNORM,
        DXGI_FORMAT_R10G10B10A2_TYPELESS => DXGI_FORMAT_R10G10B10A2_UNORM,
        DXGI_FORMAT_R16G16B16A16_TYPELESS => DXGI_FORMAT_R16G16B16A16_FLOAT,
        DXGI_FORMAT_R11G11B10_FLOAT => DXGI_FORMAT_R11G11B10_FLOAT,
        f => f,
    }
}

/// The SRV format of a depth buffer's depth, for formats the stand-in
/// reads.
pub(crate) fn depth_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format {
        DXGI_FORMAT_D32_FLOAT | DXGI_FORMAT_R32_TYPELESS => DXGI_FORMAT_R32_FLOAT,
        DXGI_FORMAT_D32_FLOAT_S8X24_UINT | DXGI_FORMAT_R32G8X24_TYPELESS => {
            DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS
        }
        DXGI_FORMAT_D24_UNORM_S8_UINT | DXGI_FORMAT_R24G8_TYPELESS => {
            DXGI_FORMAT_R24_UNORM_X8_TYPELESS
        }
        DXGI_FORMAT_D16_UNORM | DXGI_FORMAT_R16_TYPELESS => DXGI_FORMAT_R16_UNORM,
        _ => return None,
    })
}

/// The title's image `scene` (in `scene_state`) plus `albedo` (in
/// `albedo_state`), in a texture of ours left in UNORDERED_ACCESS; both
/// inputs are back in their states when its list has run. `None` (once
/// warned) when the pass cannot be made.
pub(crate) fn compose(
    host: &Host,
    scene: &ID3D12Resource,
    scene_state: u32,
    albedo: &ID3D12Resource,
    albedo_state: u32,
    depth: Option<(&ID3D12Resource, u32)>,
) -> Option<ID3D12Resource> {
    let mut pass = PASS.lock().unwrap_or_else(|e| e.into_inner());
    if pass.is_none() {
        *pass = Some(compile(host).map_err(|e| {
            tracing::warn!(error = %e, "flat light: no stand-in lighting");
        }));
    }
    let p = pass.as_mut()?.as_mut().ok()?;
    let (strength, sky, reversed) = settings();
    // SAFETY: live resources; our heap, list and output.
    unsafe {
        let sd = scene.GetDesc();
        let ad = albedo.GetDesc();
        let (w, h) = (sd.Width as u32, sd.Height);
        let (aw, ah) = (ad.Width as u32, ad.Height);
        if w == 0 || h == 0 || aw == 0 || ah == 0 {
            return None;
        }
        if p.out.as_ref().is_none_or(|o| (o.1, o.2) != (w, h)) {
            let mut out: Option<ID3D12Resource> = None;
            let made = host.device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width: u64::from(w),
                    Height: h,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    ..Default::default()
                },
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                None,
                &mut out,
            );
            match (made, out) {
                (Ok(()), Some(out)) => p.out = Some((out, w, h)),
                (made, _) => {
                    tracing::warn!(?made, w, h, "flat light: no output texture");
                    return None;
                }
            }
        }
        let out = p.out.as_ref()?.0.clone();
        let cpu = p.heap.GetCPUDescriptorHandleForHeapStart();
        let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: cpu.ptr + (i * p.step) as usize,
        };
        // An integer target (material IDs) is read at t3 and coloured from
        // a palette; t1 then holds a null view, and the other way round.
        let ids = matches!(
            ad.Format,
            DXGI_FORMAT_R8_UINT | DXGI_FORMAT_R16_UINT | DXGI_FORMAT_R32_UINT
        );
        let null = |format, i: u32| {
            host.device.CreateShaderResourceView(
                None,
                Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: format,
                    ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2D: D3D12_TEX2D_SRV {
                            MipLevels: 1,
                            ..Default::default()
                        },
                    },
                }),
                at(i),
            );
        };
        if ids {
            null(DXGI_FORMAT_R8G8B8A8_UNORM, 1);
        } else {
            null(DXGI_FORMAT_R8_UINT, 3);
        }
        for (i, (texture, desc)) in [(scene, &sd), (albedo, &ad)].into_iter().enumerate() {
            let i = if i == 1 && ids { 3 } else { i };
            host.device.CreateShaderResourceView(
                texture,
                Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: readable(desc.Format),
                    ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2D: D3D12_TEX2D_SRV {
                            MipLevels: 1,
                            ..Default::default()
                        },
                    },
                }),
                at(i as u32),
            );
        }
        let tex2d = |format| D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MipLevels: 1,
                    ..Default::default()
                },
            },
        };
        let depth = depth.and_then(|(t, state)| {
            let format = depth_format(t.GetDesc().Format)?;
            Some((t, state, format))
        });
        match depth {
            Some((t, _, format)) => {
                host.device
                    .CreateShaderResourceView(t, Some(&tex2d(format)), at(2));
            }
            // A null view: the shader does not read it.
            None => host.device.CreateShaderResourceView(
                None,
                Some(&tex2d(DXGI_FORMAT_R32_FLOAT)),
                at(2),
            ),
        }
        host.device.CreateUnorderedAccessView(&out, None, None, at(4));
        p.list.reset();
        let l = &p.list.list;
        let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        let moves: Vec<(&ID3D12Resource, D3D12_RESOURCE_STATES)> = [
            Some((scene, D3D12_RESOURCE_STATES(scene_state as i32))),
            Some((albedo, D3D12_RESOURCE_STATES(albedo_state as i32))),
            depth.map(|(t, state, _)| (t, D3D12_RESOURCE_STATES(state as i32))),
        ]
        .into_iter()
        .flatten()
        .filter(|&(_, s)| s != read && (s.0 & read.0) != read.0)
        .collect();
        let there: Vec<_> = moves.iter().map(|&(t, s)| transition(t, s, read)).collect();
        if !there.is_empty() {
            l.ResourceBarrier(&there);
        }
        l.SetDescriptorHeaps(&[Some(p.heap.clone())]);
        l.SetComputeRootSignature(&p.root);
        l.SetPipelineState(&p.pso);
        let constants = [
            w,
            h,
            aw,
            ah,
            strength.to_bits(),
            sky.to_bits(),
            u32::from(depth.is_some()),
            u32::from(reversed),
            u32::from(ids),
        ];
        l.SetComputeRoot32BitConstants(0, 9, constants.as_ptr().cast(), 0);
        l.SetComputeRootDescriptorTable(1, p.heap.GetGPUDescriptorHandleForHeapStart());
        l.Dispatch(w.div_ceil(8), h.div_ceil(8), 1);
        let back: Vec<_> = moves.iter().map(|&(t, s)| transition(t, read, s)).collect();
        let mut after = vec![D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    // SAFETY: borrowed for the call; never released here.
                    pResource: std::mem::ManuallyDrop::new(std::mem::transmute_copy(&out)),
                }),
            },
        }];
        after.extend(back);
        l.ResourceBarrier(&after);
        p.list.close();
        host.execute(&[&p.list]);
        Some(out)
    }
}
