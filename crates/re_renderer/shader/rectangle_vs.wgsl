#import <./rectangle.wgsl>

@vertex
fn vs_main(@builtin(vertex_index) v_idx: u32) -> VertexOut {
    let texcoord = Vec2(f32(v_idx / 2u), f32(v_idx % 2u));
    let pos = texcoord.x * rect_info.extent_u + texcoord.y * rect_info.extent_v +
        rect_info.top_left_corner_position;

    var out: VertexOut;
    out.position = apply_depth_offset(frame.projection_from_world * Vec4(pos, 1.0), rect_info.depth_offset);
    // out.texcoord = (texcoord.x * rect_info.extent_u + texcoord.y * rect_info.extent_v).xy;
    out.texcoord = texcoord;
    if rect_info.sample_type == SAMPLE_TYPE_NV12 || rect_info.sample_type == SAMPLE_TYPE_YUV420P {
        out.texcoord.y *= (2.0 / 3.0);
    }
    return out;
}
