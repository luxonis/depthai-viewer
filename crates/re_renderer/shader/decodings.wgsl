#import <./types.wgsl>


// fn decode_nv12(texture: texture_2d<f32>, in_tex_coords: vec2<f32>) -> Vec4 {
//     let texture_dim = Vec2(textureDimensions(texture).xy);
//     let uv_offset = u32(texture_dim.y / 1.5);
//     let uv_row = u32(floor(in_tex_coords.y * texture_dim.y) / 2.0);
//     let uv_col = u32(floor(in_tex_coords.x * texture_dim.x / 2.0) * 2.0); // 2.0 because we need two pixels for one UV pair
//     let tex_coords = UVec2(in_tex_coords * Vec2(texture_dim.x, texture_dim.y));
//     let y = textureLoad(texture, tex_coords, 0).r;
//     let u = textureLoad(texture, UVec2(uv_col, uv_offset + uv_row), 0).r;
//     let v = textureLoad(texture, UVec2((uv_col + 1u), uv_offset + uv_row), 0).r;
//     // let r = y.r + 1.13983 * (v.r - 0.5);
//     // let g = y.r - 0.39465 * (u.r - 0.5) - 0.58060 * (v.r - 0.5);
//     // let b = y.r + 2.03211 * (u.r - 0.5);
//     let r = 1.164 * (y - 0.0625) + 1.596 * (v - 0.5);
//     let g = 1.164 * (y - 0.0625) - 0.183 * (v - 0.5) - 0.391 * (u - 0.5);
//     let b = 1.164 * (y - 0.0625) + 1.596 * (u - 0.5);
//     return Vec4(y, 1.0, 1.0, 1.0);
// }

fn decode_nv12(texture: texture_2d<f32>, in_tex_coords: Vec2) -> Vec4 {
    let texture_dim = Vec2(textureDimensions(texture).xy);
    let uv_offset = u32(texture_dim.y / 1.5);
    let uv_row = u32(floor(in_tex_coords.y * texture_dim.y) / 2.0);
    let uv_col = u32(floor(in_tex_coords.x * texture_dim.x / 2.0) * 2.0); // 2.0 because we need two pixels for one UV pair
    let tex_coords = Vec2(in_tex_coords * Vec2(texture_dim.x, texture_dim.y));
    let coords = UVec2(u32(tex_coords.x), u32(tex_coords.y));
    let y = textureLoad(texture, coords, 0).r;
    let u = textureLoad(texture, UVec2(uv_col, uv_offset + uv_row), 0).r;
    let v = textureLoad(texture, UVec2((uv_col + 1u), uv_offset + uv_row), 0).r;
    // let r = y.r + 1.13983 * (v.r - 0.5);
    // let g = y.r - 0.39465 * (u.r - 0.5) - 0.58060 * (v.r - 0.5);
    // let b = y.r + 2.03211 * (u.r - 0.5);
    // let r = 1.164 * (y - 0.0625) + 1.596 * (v - 0.5);
    // let g = 1.164 * (y - 0.0625) - 0.183 * (v - 0.5) - 0.391 * (u - 0.5);
    // let b = 1.164 * (y - 0.0625) + 1.596 * (u - 0.5);

    let r = y + 1.402 * (v - 0.5);
    let g = y - 0.344136 * (u - 0.5) - 0.714136 * (v - 0.5);
    let b = y + 1.772 * (u - 0.5);
    return Vec4(r, g, b, 1.0);
}
