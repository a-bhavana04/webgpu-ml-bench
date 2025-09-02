enable f16;

struct X { data: array<f16>; };
struct Y { data: array<f16>; };

@group(0) @binding(0) var<storage, read>       IN  : X;
@group(0) @binding(1) var<storage, read_write> OUT : Y;
@group(0) @binding(2) var<uniform>             dims: vec2<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let L = dims.x * dims.y;
  let i = gid.x;
  if (i >= L) { return; }
  let x : f16 = IN.data[i];
  let y : f16 = 0.5h * x * (1.0h + tanh(0.79788456h * (x + 0.044715h * x * x * x)));
  OUT.data[i] = y;
}
