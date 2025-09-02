import * as ort from "onnxruntime-web";
import type { BenchResult } from "./types";

export async function benchONNX(
  modelUrl: string,
  inputShape: number[],
  repeats = 5
): Promise<BenchResult> {
  ort.env.wasm.wasmPaths = "/ort-wasm/";
  ort.env.debug = false;

  const session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ["webgpu" as any],
    graphOptimizationLevel: "all"
  });

  const size = inputShape.reduce((a, b) => a * b, 1);
  const input = new Float32Array(size).fill(1);
  const feeds: Record<string, ort.Tensor> = {};

  const firstInput = session.inputNames[0] ?? "input";
  feeds[firstInput] = new ort.Tensor("float32", input, inputShape);

  const times: number[] = [];
  await session.run(feeds);
  for (let i = 0; i < repeats; i++) {
    const t0 = performance.now();
    await session.run(feeds);
    const t1 = performance.now();
    times.push(t1 - t0);
  }

  const sorted = times.slice().sort((a, b) => a - b);
  const p50 = sorted[Math.floor(repeats * 0.5)];
  const p95 = sorted[Math.floor(repeats * 0.95)];
  const avg = times.reduce((a, b) => a + b, 0) / repeats;
  const throughput = 1 / (p50 / 1000);

  return {
    op: "onnx_model",
    vendor: "onnxruntime-web",
    dtype: "f32",
    time: p50,
    p50,
    p95,
    avg,
    size: 1,
    shape: { input: inputShape.join("x") as any },
    useGpuTs: false,
    gbps: throughput
  };
}
