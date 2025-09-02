import { initGPU } from "../util";
import { BenchResult } from "./types";
import gemvF32 from "../kernels/gemv_f32.wgsl?raw";
import gemvF16 from "../kernels/gemv_f16.wgsl?raw";

export async function benchGEMV(M: number, K: number, dtype: "f32" | "f16", repeats = 10): Promise<BenchResult> {
  const { device, vendor, features } = await initGPU();
  const useF16 = dtype === "f16";
  if (useF16 && !(features as Set<GPUFeatureName>).has("shader-f16")) {
    throw new Error("shader-f16 not supported on this device");
  }

  const elemSize = useF16 ? 2 : 4;
  const A = new Float32Array(M * K).fill(1).map(() => Math.random());
  const v = new Float32Array(K).fill(1).map(() => Math.random());
  const ABuffer = device.createBuffer({ size: A.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const vBuffer = device.createBuffer({ size: v.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(ABuffer, 0, A.buffer);
  device.queue.writeBuffer(vBuffer, 0, v.buffer);

  const yBuffer = device.createBuffer({ size: M * elemSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const dimsBuffer = device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(dimsBuffer, 0, new Uint32Array([M, K]));

  const module = device.createShaderModule({ code: useF16 ? gemvF16 : gemvF32 });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: ABuffer } },
      { binding: 1, resource: { buffer: vBuffer } },
      { binding: 2, resource: { buffer: yBuffer } },
      { binding: 3, resource: { buffer: dimsBuffer } }
    ]
  });

  const times: number[] = [];
  for (let i = 0; i < repeats; i++) {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(M);
    pass.end();
    const t0 = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const t1 = performance.now();
    times.push(t1 - t0);
  }

  const sorted = times.slice().sort((a, b) => a - b);
  const p50 = sorted[Math.floor(repeats * 0.5)];
  const p95 = sorted[Math.floor(repeats * 0.95)];
  const avg = times.reduce((a, b) => a + b, 0) / repeats;
  const gbps = ((M * K + K + M) * elemSize) / (p50 * 1e6);

  return {
  op: "gemv",
  vendor,
  useGpuTs: false,
  dtype,
  time: p50,                 
  p50,
  p95,
  avg,
  size: M * K,               
  shape: { size: M * K, M, K },
  gbps,
};

}
