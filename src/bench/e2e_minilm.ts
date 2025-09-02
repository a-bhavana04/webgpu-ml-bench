import { initGPU } from "../util";
import { BenchResult } from "./types";
import { pipeline, env } from "@xenova/transformers";

export async function benchMiniLMEmbeddings(
  texts: string[],
  dtype: "f32" | "f16" = "f32",
  repeats = 5
): Promise<BenchResult> {
  const { vendor } = await initGPU();
  env.allowLocalModels = false;

const extractor = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2",
  { device: "webgpu" } as unknown as any
);


  await extractor(texts);

  const times: number[] = [];
  for (let i = 0; i < repeats; i++) {
    const t0 = performance.now();
    await extractor(texts);
    const t1 = performance.now();
    times.push(t1 - t0);
  }

  const sorted = times.slice().sort((a, b) => a - b);
  const p50 = sorted[Math.floor(repeats * 0.5)];
  const p95 = sorted[Math.floor(repeats * 0.95)];
  const avg = times.reduce((a, b) => a + b, 0) / repeats;
  const throughput = texts.length / (p50 / 1000);

  return {
    op: "minilm_embed",
    vendor,
    dtype,
    time: p50,
    p50,
    p95,
    avg,
    size: texts.length,
    shape: { batch: texts.length },
    useGpuTs: false,
    gbps: throughput,
  };
}
