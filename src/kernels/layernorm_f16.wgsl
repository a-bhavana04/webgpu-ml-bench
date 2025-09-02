enable f16;
struct X { data: array<f16>; };
struct Y { data: array<f16>; };
@group(0) @binding(0) var<storage, read> IN : X;
@group(0) @binding(1) var<storage, read_write> OUT : Y;
@group(0) @binding(2) var<uniform> dims : vec3<u32>;
@group(0) @binding(3) var<uniform> eps : f16;
@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let B = dims.x;
  let N = dims.y;
  let b = gid.x;
  if (b >= B) { return; }
  var mean = 0.0h;
  for (var i = 0u; i < N; i = i + 1u) {
    mean = mean + IN.data[b*N + i];
  }
  mean = mean / f16(N);
  var varsum = 0.0h;
  for (var i = 0u; i < N; i = i + 1u) {
    let diff = IN.data[b*N + i] - mean;
    varsum = varsum + diff * diff;
  }
  let variance = varsum / f16(N);
  let denom = sqrt(variance + eps);
  for (var i = 0u; i < N; i = i + 1u) {
    OUT.data[b*N + i] = (IN.data[b*N + i] - mean) / denom;
  }
}
