override TM : u32 = 16u;
override TN : u32 = 16u;
override TK : u32 = 16u;

struct Matrix { data: array<f32>, };
struct Dims { M: u32, N: u32, K: u32, _pad: u32, };

@group(0) @binding(0) var<storage, read>       A : Matrix;
@group(0) @binding(1) var<storage, read>       B : Matrix;
@group(0) @binding(2) var<storage, read_write> C : Matrix;
@group(0) @binding(3) var<uniform>             dims : Dims;

var<workgroup> tileA : array<f32, TM * TK>;
var<workgroup> tileB : array<f32, TK * TN>;

@compute @workgroup_size(TN, TM, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_id)  lid : vec3<u32>) {
  let M = dims.M; let N = dims.N; let K = dims.K;
  let row = gid.y; let col = gid.x;

  var acc : f32 = 0.0;
  var kBase : u32 = 0u;

  loop {
    if (kBase >= K) { break; }

    var kkA : u32 = lid.x;
    loop {
      if (kkA >= TK) { break; }
      let aCol = kBase + kkA;
      var aVal : f32 = 0.0;
      if (row < M && aCol < K) { aVal = A.data[row * K + aCol]; }
      tileA[lid.y * TK + kkA] = aVal;
      kkA += TN;
    }

    var kkB : u32 = lid.y;
    loop {
      if (kkB >= TK) { break; }
      let bRow = kBase + kkB;
      var bVal : f32 = 0.0;
      if (bRow < K && col < N) { bVal = B.data[bRow * N + col]; }
      tileB[kkB * TN + lid.x] = bVal;
      kkB += TM;
    }

    workgroupBarrier();

    for (var k : u32 = 0u; k < TK; k = k + 1u) {
      acc = acc + tileA[lid.y * TK + k] * tileB[k * TN + lid.x];
    }

    workgroupBarrier();
    kBase = kBase + TK;
  }

  if (row < M && col < N) {
    C.data[row * N + col] = acc;
  }
}
