struct A { data: array<f32>; };
struct v { data: array<f32>; };
struct y { data: array<f32>; };
@group(0) @binding(0) var<storage, read> IN_A : A;
@group(0) @binding(1) var<storage, read> IN_v : v;
@group(0) @binding(2) var<storage, read_write> OUT_y : y;
@group(0) @binding(3) var<uniform> dims : vec2<u32>;
@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let M = dims.x;
  let K = dims.y;
  let m = gid.x;
  if (m >= M) { return; }
  var acc = 0.0;
  for (var k = 0u; k < K; k = k + 1u) {
    acc = acc + IN_A.data[m*K + k] * IN_v.data[k];
  }
  OUT_y.data[m] = acc;
}
