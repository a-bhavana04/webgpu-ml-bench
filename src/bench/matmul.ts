import { initGPU } from "../util";
import matmulF32 from "../kernels/matmul_f32.wgsl?raw";
import matmulF16 from "../kernels/matmul_f16.wgsl?raw";

export type MatmulResult = {
  op: string;
  vendor: string;
  useGpuTs: boolean;
  shape: { M: number; N: number; K: number };
  tiles: { tm: number; tn: number; tk: number };
  dtype: "f32" | "f16";
  p50: number; p95: number; avg: number;
  gflops: number;
  first: number; expect: number;
};

type Opts = { tm?: number; tn?: number; tk?: number; repeats?: number };

function stats(ms: number[]) {
  const a = [...ms].sort((x, y) => x - y);
  const q = (p: number) => a[Math.min(a.length - 1, Math.floor(p * (a.length - 1)))];
  return { p50: q(0.5), p95: q(0.95), avg: a.reduce((s, x) => s + x, 0) / a.length };
}

function f16bits(x: number) {
  const s = x < 0 ? 1 : 0; const ax = Math.abs(x);
  if (!Number.isFinite(ax)) return (s << 15) | 0x7c00;
  if (ax === 0) return s << 15;
  let e = Math.floor(Math.log2(ax));
  let m = ax / Math.pow(2, e) - 1;
  e += 15;
  if (e <= 0) return s << 15;
  if (e >= 31) return (s << 15) | 0x7c00;
  return (s << 15) | (e << 10) | Math.floor(m * 1024 + 0.5);
}

function f16toF32(h: number) {
  const s = (h >> 15) & 1; let e = (h >> 10) & 0x1f; let m = h & 0x3ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (m / 1024);
  if (e === 31) return m ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + m / 1024);
}

async function bench(code: string, M: number, N: number, K: number, dtype: "f32" | "f16", opts: Opts) {
  const { device, features, vendor } = await initGPU();
  const tm = opts.tm ?? 16, tn = opts.tn ?? 16, tk = opts.tk ?? 16;
  const repeats = Math.max(1, opts.repeats ?? 10);

  const aBytes = dtype === "f16" ? 2 : 4;
  const bytesA = M * K * aBytes, bytesB = K * N * aBytes, bytesC = M * N * aBytes;

  let bufA: GPUBuffer, bufB: GPUBuffer;
  if (dtype === "f16") {
    const A = new Uint16Array(M * K); const B = new Uint16Array(K * N);
    A.fill(f16bits(1)); B.fill(f16bits(1));
    bufA = device.createBuffer({ size: bytesA, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    bufB = device.createBuffer({ size: bytesB, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(bufA, 0, A);
    device.queue.writeBuffer(bufB, 0, B);
  } else {
    const A = new Float32Array(M * K).fill(1);
    const B = new Float32Array(K * N).fill(1);
    bufA = device.createBuffer({ size: bytesA, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    bufB = device.createBuffer({ size: bytesB, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(bufA, 0, A);
    device.queue.writeBuffer(bufB, 0, B);
  }

  const bufC = device.createBuffer({ size: bytesC, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const dims = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(dims, 0, new Uint32Array([M, N, K, 0]));

  const module = device.createShaderModule({ code });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main", constants: { TM: tm, TN: tn, TK: tk } }
  });
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufA } },
      { binding: 1, resource: { buffer: bufB } },
      { binding: 2, resource: { buffer: bufC } },
      { binding: 3, resource: { buffer: dims } }
    ]
  });

  const dx = Math.ceil(N / tn);
  const dy = Math.ceil(M / tm);
  const useGpuTs = (features as Set<GPUFeatureName>).has("timestamp-query");
  const times: number[] = [];

  {
    const e = device.createCommandEncoder();
    const p = e.beginComputePass();
    p.setPipeline(pipeline); p.setBindGroup(0, bind); p.dispatchWorkgroups(dx, dy, 1); p.end();
    device.queue.submit([e.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  for (let i = 0; i < repeats; i++) {
    let gpuMs: number | null = null;
    const enc = device.createCommandEncoder();
    let qs: GPUQuerySet | null = null; let qb: GPUBuffer | null = null;
    let desc: GPUComputePassDescriptor = {};
    if (useGpuTs) {
      qs = device.createQuerySet({ type: "timestamp", count: 2 });
      qb = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.QUERY_RESOLVE });
      desc = { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } };
    }
    const pass = enc.beginComputePass(desc);
    pass.setPipeline(pipeline); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(dx, dy, 1); pass.end();
    if (qs && qb) enc.resolveQuerySet(qs, 0, 2, qb, 0);

    const t0 = performance.now();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    const t1 = performance.now();

    if (qb) {
      const rb = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const e2 = device.createCommandEncoder();
      e2.copyBufferToBuffer(qb, 0, rb, 0, 16);
      device.queue.submit([e2.finish()]);
      await rb.mapAsync(GPUMapMode.READ);
      const ns = new BigUint64Array(rb.getMappedRange());
      gpuMs = Number(ns[1] - ns[0]) / 1e6;
      rb.unmap();
    }
    times.push(gpuMs ?? (t1 - t0));
  }

  const s = stats(times);
  const gflops = (2 * M * N * K) / (s.p50 * 1e6);

  const readSize = 4; 
  const read = device.createBuffer({ size: readSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  {
    const e = device.createCommandEncoder();
    e.copyBufferToBuffer(bufC, 0, read, 0, readSize);
    device.queue.submit([e.finish()]);
    await read.mapAsync(GPUMapMode.READ);
  }
  let first = 0;
  if (dtype === "f16") {
    const u16 = new Uint16Array(read.getMappedRange().slice(0, 2));
    first = f16toF32(u16[0]);
  } else {
    const f32 = new Float32Array(read.getMappedRange().slice(0, 4));
    first = f32[0];
  }
  read.unmap();

  return {
    op: "matmul",
    vendor, useGpuTs,
    shape: { M, N, K }, tiles: { tm, tn, tk }, dtype,
    p50: s.p50, p95: s.p95, avg: s.avg, gflops,
    first, expect: K
  };
}

export async function benchMatmulF32(M: number, N: number, K: number, opts: Opts = {}) {
  return bench(matmulF32, M, N, K, "f32", opts);
}

export async function benchMatmulF16(M: number, N: number, K: number, opts: Opts = {}) {
  return bench(matmulF16, M, N, K, "f16", opts);
}
