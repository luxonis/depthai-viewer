//! Renders a point cloud from a depth texture and a set of intrinsics.
//!
//! See `src/renderer/depth_cloud.rs` for more documentation.

#import <./colormap.wgsl>
#import <./global_bindings.wgsl>
#import <./types.wgsl>
#import <./utils/camera.wgsl>
#import <./utils/flags.wgsl>
#import <./utils/size.wgsl>
#import <./utils/sphere_quad.wgsl>
#import <./utils/srgb.wgsl>
#import <./decodings.wgsl>

const SAMPLE_TYPE_FLOAT_FILTER   = 1u;
const SAMPLE_TYPE_FLOAT_NOFILTER = 2u;
const SAMPLE_TYPE_SINT_NOFILTER  = 3u;
const SAMPLE_TYPE_UINT_NOFILTER  = 4u;
// -------------------
//  Encoded textures
// -------------------
const SAMPLE_TYPE_NV12  = 5u;

// ---

/// Keep in sync with `DepthCloudInfoUBO` in `depth_cloud.rs`.
///
/// Same for all draw-phases.
struct DepthCloudInfo {
    /// The extrinsincs of the camera used for the projection.
    world_from_obj: Mat4,

    /// The intrinsics of the camera used for the projection.
    ///
    /// Only supports pinhole cameras at the moment.
    depth_camera_intrinsics: Mat3,

    /// Outline mask id for the outline mask pass.
    outline_mask_id: UVec2,

    /// Picking object id that applies for the entire depth cloud.
    picking_layer_object_id: UVec2,

    /// Multiplier to get world-space depth from whatever is in the texture.
    world_depth_from_texture_value: f32,

    /// Point radius is calculated as world-space depth times this value.
    point_radius_from_world_depth: f32,

    /// The maximum depth value in world-space, for use with the colormap.
    max_depth_in_world: f32,

    /// Configures color mapping mode, see `colormap.wgsl`.
    colormap: u32,

    /// Configures the sample type for the albedo texture.
    albedo_sample_type: u32,

    /// Uint or filterable float.
    depth_sample_type: u32,

    /// Changes between the opaque and outline draw-phases.
    radius_boost_in_ui_points: f32,
};

const ALBEDO_COLOR_RGB: u32 = 0u;
const ALBEDO_COLOR_MONO: u32 = 1u;

@group(1) @binding(0)
var<uniform> depth_cloud_info: DepthCloudInfo;

@group(1) @binding(1)
var depth_texture_float: texture_2d<f32>;

@group(1) @binding(2)
var depth_texture_uint: texture_2d<u32>;

// -----------------------------------------------
//  Different kinds of albedo textures
//  Only sampled when colormap == ALBEDO_TEXTURE
// -----------------------------------------------
@group(1) @binding(3)
var albedo_texture_float_nofilter: texture_2d<f32>;

@group(1) @binding(4)
var albedo_texture_sint: texture_2d<i32>;

@group(1) @binding(5)
var albedo_texture_uint: texture_2d<u32>;

@group(1) @binding(6)
var albedo_texture_float_filterable: texture_2d<f32>;


struct VertexOut {
    @builtin(position)
    pos_in_clip: Vec4,

    @location(0) @interpolate(perspective)
    pos_in_world: Vec3,

    @location(1) @interpolate(flat)
    point_pos_in_world: Vec3,

    @location(2) @interpolate(flat)
    point_color: Vec4,

    @location(3) @interpolate(flat)
    point_radius: f32,

    @location(4) @interpolate(flat)
    quad_idx: u32,
};

// ---

struct PointData {
    pos_in_world: Vec3,
    unresolved_radius: f32,
    color: Vec4,
}

// Backprojects the depth texture using the intrinsics passed in the uniform buffer.
fn compute_point_data(quad_idx: u32) -> PointData {
    var wh: UVec2;
    var texcoords: UVec2;
    var world_space_depth: f32 = depth_cloud_info.world_depth_from_texture_value;
    if depth_cloud_info.depth_sample_type == SAMPLE_TYPE_FLOAT_FILTER {
        wh = textureDimensions(depth_texture_float);
        // TODO(cmc): expose knobs to linearize/normalize/flip/cam-to-plane depth.
        texcoords = UVec2(quad_idx % wh.x, quad_idx / wh.x);
        world_space_depth = world_space_depth * textureLoad(depth_texture_float, texcoords, 0).x;
    } else {
        wh = textureDimensions(depth_texture_uint);
        texcoords = UVec2(quad_idx % wh.x, quad_idx / wh.x);
        world_space_depth = world_space_depth * f32(textureLoad(depth_texture_uint, texcoords, 0).x);
    }

    var data: PointData;
    if 0.0 < world_space_depth && world_space_depth < f32max {
        // TODO(cmc): albedo textures
        // let color = Vec4(colormap_linear(depth_cloud_info.colormap, world_space_depth / depth_cloud_info.max_depth_in_world), 1.0);

        var color: Vec4;
        if depth_cloud_info.colormap == ALBEDO_TEXTURE {
            if depth_cloud_info.albedo_sample_type == SAMPLE_TYPE_NV12 {
                color = decode_nv12(albedo_texture_uint, Vec2(f32(texcoords.x), f32(texcoords.y)) / Vec2(f32(wh.x), f32(wh.x)));
            } else { // TODO(filip): Support all sample types like in rectangle_fs.wgsl
                if depth_cloud_info.albedo_sample_type == SAMPLE_TYPE_FLOAT_FILTER {
                    color = textureSampleLevel(
                        albedo_texture_float_filterable,
                        trilinear_sampler,
                        Vec2(texcoords) / Vec2(wh),
                        0.0
                    );
                    color = Vec4(Vec3(color.r), 1.0);
                } else if depth_cloud_info.albedo_sample_type == SAMPLE_TYPE_FLOAT_NOFILTER {
                    let coord = Vec2(texcoords) / Vec2(textureDimensions(depth_texture_float));
                    let v00 = textureLoad(albedo_texture_float_nofilter, IVec2(coord) + IVec2(0, 0), 0);
                    let v01 = textureLoad(albedo_texture_float_nofilter, IVec2(coord) + IVec2(0, 1), 0);
                    let v10 = textureLoad(albedo_texture_float_nofilter, IVec2(coord) + IVec2(1, 0), 0);
                    let v11 = textureLoad(albedo_texture_float_nofilter, IVec2(coord) + IVec2(1, 1), 0);
                    let top = mix(v00, v10, fract(coord.x));
                    let bottom = mix(v01, v11, fract(coord.x));
                    color = mix(top, bottom, fract(coord.y));
                } else if depth_cloud_info.albedo_sample_type == SAMPLE_TYPE_UINT_NOFILTER {
                    color = Vec4(textureLoad(
                        albedo_texture_uint,
                        texcoords,
                        0
                    )) / 255.0;
                    color = Vec4(linear_from_srgb(Vec3(color.r)), 1.0);
                } else if depth_cloud_info.albedo_sample_type == SAMPLE_TYPE_SINT_NOFILTER {
                    color = Vec4(textureLoad(
                        albedo_texture_sint,
                        texcoords,
                        0
                    )) / 255.0;
                    color = Vec4(linear_from_srgb(Vec3(color.r)), 1.0);
                } else {
                    color = ERROR_RGBA;
                }
              }
        } else {
            color = Vec4(colormap_srgb(depth_cloud_info.colormap, world_space_depth), 1.0);
        }
        // TODO(cmc): This assumes a pinhole camera; need to support other kinds at some point.
        let intrinsics = depth_cloud_info.depth_camera_intrinsics;
        let focal_length = Vec2(intrinsics[0][0], intrinsics[1][1]);
        let offset = Vec2(intrinsics[2][0], intrinsics[2][1]);

        let pos_in_obj = Vec3(
            (Vec2(texcoords) - offset) * world_space_depth / focal_length,
            world_space_depth,
        );

        let pos_in_world = depth_cloud_info.world_from_obj * Vec4(pos_in_obj, 1.0);

        data.pos_in_world = pos_in_world.xyz;
        data.unresolved_radius = depth_cloud_info.point_radius_from_world_depth * world_space_depth;
        data.color = color;
    } else {
        // Degenerate case
        data.pos_in_world = Vec3(0.0);
        data.unresolved_radius = 0.0;
        data.color = Vec4(0.0);
    }
    return data;
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_idx: u32) -> VertexOut {
    let quad_idx = sphere_quad_index(vertex_idx);

    // Compute point data (valid for the entire quad).
    let point_data = compute_point_data(quad_idx);

    var out: VertexOut;
    out.point_pos_in_world = point_data.pos_in_world;
    out.point_color = point_data.color;
    out.quad_idx = u32(quad_idx);

    if 0.0 < point_data.unresolved_radius {
        // Span quad
        let quad = sphere_quad_span(vertex_idx, point_data.pos_in_world, point_data.unresolved_radius, depth_cloud_info.radius_boost_in_ui_points);
        out.pos_in_clip = frame.projection_from_world * Vec4(quad.pos_in_world, 1.0);
        out.pos_in_world = quad.pos_in_world;
        out.point_radius = quad.point_resolved_radius;
    } else {
        // Degenerate case - early-out!
        out.pos_in_clip = Vec4(0.0);
        out.pos_in_world = Vec3(0.0);
        out.point_radius = 0.0;
    }

    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) Vec4 {
    let coverage = sphere_quad_coverage(in.pos_in_world, in.point_radius, in.point_pos_in_world);
    if coverage < 0.001 {
        discard;
    }
    return vec4(in.point_color.rgb, coverage);
}

@fragment
fn fs_main_picking_layer(in: VertexOut) -> @location(0) UVec4 {
    let coverage = sphere_quad_coverage(in.pos_in_world, in.point_radius, in.point_pos_in_world);
    if coverage <= 0.5 {
        discard;
    }
    return UVec4(depth_cloud_info.picking_layer_object_id, in.quad_idx, 0u);
}

@fragment
fn fs_main_outline_mask(in: VertexOut) -> @location(0) UVec2 {
    // TODO(filip): Proper outline mask for a single point?
    // Output is an integer target, can't use coverage therefore.
    // But we still want to discard fragments where coverage is low.
    // Since the outline extends a bit, a very low cut off tends to look better.
    let coverage = sphere_quad_coverage(in.pos_in_world, in.point_radius, in.point_pos_in_world);
    if coverage <= 1.0 {
        discard;
    }
    return depth_cloud_info.outline_mask_id;
}
