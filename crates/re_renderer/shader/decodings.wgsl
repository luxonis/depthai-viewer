#import <./types.wgsl>

fn decode_nv12(texture: texture_2d<u32>, in_tex_coords: Vec2) -> Vec4 {
    let texture_dim = Vec2(textureDimensions(texture).xy);
    let uv_offset = u32(floor(texture_dim.y / 1.5));
    let uv_row = u32(floor(in_tex_coords.y * texture_dim.y) / 2.0);
    var uv_col = u32(floor(in_tex_coords.x * texture_dim.x / 2.0)) * 2u;

    let coords = UVec2(in_tex_coords * Vec2(texture_dim.x, texture_dim.y));
    let y = (f32(textureLoad(texture, coords, 0).r) - 16.0) / 219.0;
    let u = (f32(textureLoad(texture, UVec2(u32(uv_col), uv_offset + uv_row), 0).r) - 128.0) / 224.0;
    let v = (f32(textureLoad(texture, UVec2((u32(uv_col) + 1u), uv_offset + uv_row), 0).r) - 128.0) / 224.0;

    // Get RGB values and apply reverse gamma correction since we are rendering to sRGB framebuffer
    let r = pow(y + 1.402 * v, 2.2);
    let g = pow(y  - (0.344 * u + 0.714 * v), 2.2);
    let b = pow(y + 1.772 * u, 2.2);
    return Vec4(r, g, b, 1.0);
}
