import { initGPU } from "../util";
import { BenchResult } from "./types";
import layernormF32 from "../kernels/layernorm_f32.wgsl?raw";
import layernormF16 from "../kernels/layernorm_f16.wgsl?raw";

export async function benchLayerNorm(B: number, N: number, useAffine: boolean, eps: number, dtype: "f32" | "f16", repeats = 10): Promise<BenchResult> {
  const { device, vendor, features } = await initGPU();
  const useF16 = dtype === "f16";
  if (useF16 && !(features as Set<GPUFeatureName>).has("shader-f16")) {
    throw new Error("shader-f16 not supported on this device");
  }

  const elemSize = useF16 ? 2 : 4;
  const input = new Float32Array(B * N).fill(1).map(() => Math.random());
  const inputBuffer = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(inputBuffer, 0, input.buffer);

  const outputBuffer = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const dimsBuffer = device.createBuffer({ size: 12, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(dimsBuffer, 0, new Uint32Array([B, N, useAffine ? 1 : 0]));
  const epsBuffer = device.createBuffer({ size: elemSize, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(epsBuffer, 0, useF16 ? new Uint16Array([eps]) : new Float32Array([eps]));

  const module = device.createShaderModule({ code: useF16 ? layernormF16 : layernormF32 });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
      { binding: 2, resource: { buffer: dimsBuffer } },
      { binding: 3, resource: { buffer: epsBuffer } }
    ]
  });

  const times: number[] = [];
  for (let i = 0; i < repeats; i++) {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(B);
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
  const gbps = (5 * B * N * elemSize) / (p50 * 1e6);

  return {
  op: "layernorm",
  vendor,
  useGpuTs: false,
  dtype,
  time: p50,
  p50,
  p95,
  avg,
  size: B * N,
  shape: { size: B * N, B, N },
  gbps,
};

}
