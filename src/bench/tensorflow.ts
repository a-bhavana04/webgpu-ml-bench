import * as tf from "@tensorflow/tfjs";
import type { BenchResult } from "./types";

export async function benchTensorFlow(modelUrl: string, inputShape: number[], repeats = 5): Promise<BenchResult> {
  const model = await tf.loadGraphModel(modelUrl);

  const size = inputShape.reduce((a, b) => a * b, 1);
  const input = tf.tensor(new Float32Array(size).fill(1), inputShape);

  const times: number[] = [];
  // warmup
  await model.predict(input);
  for (let i = 0; i < repeats; i++) {
    const t0 = performance.now();
    await model.predict(input);
    const t1 = performance.now();
    times.push(t1 - t0);
  }

  const sorted = times.slice().sort((a, b) => a - b);
  const p50 = sorted[Math.floor(repeats * 0.5)];
  const p95 = sorted[Math.floor(repeats * 0.95)];
  const avg = times.reduce((a, b) => a + b, 0) / repeats;
  const throughput = 1 / (p50 / 1000);

  return {
    op: "tensorflow_model",
    vendor: "tensorflow.js",
    dtype: "f32",
    time: p50,
    p50,
    p95,
    avg,
    size: 1,
    shape: { input: inputShape.join("x") as any },
    useGpuTs: false,
    gbps: throughput,
  };
}
