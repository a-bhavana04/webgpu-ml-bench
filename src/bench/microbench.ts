export type BenchResult = {
  op: string;
  shape: Record<string, number>;
  vendor: string;
  features: string[];
  useF16: boolean;
  gpuMs?: number;
  cpuMs: number;
  gflops?: number;
  gbps?: number;
  p50?: number;
  p95?: number;
};

// Minimal stub for matmul microbenchmark
declare const matmulF32WGSL: string;

export async function benchMatmul(M: number, N: number, K: number, repeats = 20): Promise<BenchResult> {
  // TODO: implement full microbenchmark logic
  // This is a stub to wire up the flow
  return {
    op: "matmul_f32",
    shape: { M, N, K },
    vendor: "",
    features: [],
    useF16: false,
    cpuMs: 0,
  };
}
