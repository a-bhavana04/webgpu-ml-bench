enable f16;
struct X { data: array<f16>; };
struct Y { data: array<f16>; };
@group(0) @binding(0) var<storage, read> IN : X;
@group(0) @binding(1) var<storage, read_write> OUT : Y;
@group(0) @binding(2) var<uniform> dims : vec2<u32>;
var<workgroup> row : array<f16, 1024>;
@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let B = dims.x;
  let N = dims.y;
  let b = gid.x;
  if (b >= B) { return; }
  var maxval = -1e4h;
  for (var i = 0u; i < N; i = i + 1u) {
    let v = IN.data[b*N + i];
    if (v > maxval) { maxval = v; }
  }
  var sum = 0.0h;
  for (var i = 0u; i < N; i = i + 1u) {
    let e = exp(IN.data[b*N + i] - maxval);
    row[i] = e;
    sum = sum + e;
  }
  for (var i = 0u; i < N; i = i + 1u) {
    OUT.data[b*N + i] = row[i] / sum;
  }
}
