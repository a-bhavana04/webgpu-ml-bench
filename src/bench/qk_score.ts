import { BenchResult } from "./types";
import { initGPU } from "../util";
import { createGPUBuffer, runComputePipeline } from "../util";
import qkScoreF16WGSL from "../kernels/qk_score_f16.wgsl?raw";
import qkScoreF32WGSL from "../kernels/qk_score_f32.wgsl?raw";

export async function benchQKScore(size: number, dtype: "f16" | "f32"): Promise<BenchResult> {
    const { device, vendor } = await initGPU();
    const inputBufferA = createGPUBuffer(device, size * size * Float32Array.BYTES_PER_ELEMENT, GPUBufferUsage.STORAGE);
    const inputBufferB = createGPUBuffer(device, size * size * Float32Array.BYTES_PER_ELEMENT, GPUBufferUsage.STORAGE);
    const outputBuffer = createGPUBuffer(device, size * size * Float32Array.BYTES_PER_ELEMENT, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

    const shaderCode = dtype === "f16" ? qkScoreF16WGSL : qkScoreF32WGSL;
    const pipeline = device.createComputePipeline({
        layout: "auto", 
        compute: {
            module: device.createShaderModule({ code: shaderCode }),
            entryPoint: "main",
        },
    });

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: inputBufferA } },
            { binding: 1, resource: { buffer: inputBufferB } },
            { binding: 2, resource: { buffer: outputBuffer } },
        ],
    });

    const start = performance.now();
    await runComputePipeline(device, pipeline, bindGroup, size);
    const end = performance.now();

    return {
        op: "qk_score",
        size,
        time: end - start,
        vendor,
        useGpuTs: true,
        dtype,
        shape: { size },
        p50: 0,
        p95: 0,
        avg: (end - start) / size,
    };
}
