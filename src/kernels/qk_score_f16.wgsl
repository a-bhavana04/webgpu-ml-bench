enable f16;
struct Q { data: array<f16>; };
struct Kmat { data: array<f16>; };
struct A { data: array<f16>; };
@group(0) @binding(0) var<storage, read> IN_Q : Q;
@group(0) @binding(1) var<storage, read> IN_K : Kmat;
@group(0) @binding(2) var<storage, read_write> OUT_A : A;
@group(0) @binding(3) var<uniform> dims : vec3<u32>;
@group(0) @binding(4) var<uniform> scale : f16;
@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let M = dims.x;
  let N = dims.y;
  let K = dims.z;
  let m = gid.x;
  let n = gid.y;
  if (m >= M || n >= N) { return; }
  var acc = 0.0h;
  for (var k = 0u; k < K; k = k + 1u) {
    acc = acc + IN_Q.data[m*K + k] * IN_K.data[n*K + k];
  }
  OUT_A.data[m*N + n] = acc * scale;
}
