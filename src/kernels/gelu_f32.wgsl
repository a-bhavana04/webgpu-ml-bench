struct X { data: array<f32>; };
struct Y { data: array<f32>; };

@group(0) @binding(0) var<storage, read>       IN  : X;
@group(0) @binding(1) var<storage, read_write> OUT : Y;
@group(0) @binding(2) var<uniform>             dims: vec2<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let L = dims.x * dims.y;
  let i = gid.x;
  if (i >= L) { return; }
  let x : f32 = IN.data[i];
  let y : f32 = 0.5 * x * (1.0 + tanh(0.79788456 * (x + 0.044715 * x * x * x)));
  OUT.data[i] = y;
}
