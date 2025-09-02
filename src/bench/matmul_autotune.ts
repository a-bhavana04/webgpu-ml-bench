import { benchMatmulF32, benchMatmulF16, MatmulResult } from "./matmul";

export type TuneOut = { best: MatmulResult; results: MatmulResult[] };

export async function autotuneMatmul(M: number, N: number, K: number, repeats: number, dtype: "f32" | "f16"): Promise<TuneOut> {
  const shapes = [
    { tm: 16, tn: 16, tk: 16 },
    { tm: 32, tn: 8,  tk: 16 },
    { tm: 8,  tn: 32, tk: 16 }
  ];
  const results: MatmulResult[] = [];
  for (const s of shapes) {
    const r = dtype === "f16"
      ? await benchMatmulF16(M, N, K, { ...s, repeats })
      : await benchMatmulF32(M, N, K, { ...s, repeats });
    results.push(r);
  }
  results.sort((a, b) => a.p50 - b.p50);
  return { best: results[0], results };
}
