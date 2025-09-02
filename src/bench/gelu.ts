import type { BenchResult } from "./types";
import gelu16WGSL from "../kernels/gelu_f16.wgsl?raw";
import gelu32WGSL from "../kernels/gelu_f32.wgsl?raw";
import { initGPU } from "../util";

const WGS = 256;

export async function benchGELU(elements: number, dtype: "f16" | "f32"): Promise<BenchResult> {
  const { device, vendor } = await initGPU();

  const elemBytes = dtype === "f16" ? 2 : 4;
  const inBuf = device.createBuffer({ size: elements * elemBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const outBuf = device.createBuffer({ size: elements * elemBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const dims = device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  if (dtype === "f16") {
    const ones16 = new Uint16Array(elements).fill(0x3c00);
    device.queue.writeBuffer(inBuf, 0, ones16);
  } else {
    const ones32 = new Float32Array(elements).fill(1);
    device.queue.writeBuffer(inBuf, 0, ones32);
  }
  device.queue.writeBuffer(dims, 0, new Uint32Array([1, elements]));

  const module = device.createShaderModule({ code: dtype === "f16" ? gelu16WGSL : gelu32WGSL });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inBuf } },
      { binding: 1, resource: { buffer: outBuf } },
      { binding: 2, resource: { buffer: dims } },
    ],
  });

  const groups = Math.ceil(elements / WGS);

  {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(groups);
    pass.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  const t0 = performance.now();
  {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(groups);
    pass.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
  const t1 = performance.now();

  return {
    op: "gelu",
    size: elements,
    time: t1 - t0,
    vendor,
    useGpuTs: true,
    dtype,
    shape: { size: elements, elements },
    p50: 0,
    p95: 0,
    avg: (t1 - t0) / elements,
  };
}
