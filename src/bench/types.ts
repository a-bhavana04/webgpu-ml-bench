export interface BenchResult {
  op: string;
  vendor: string;
  dtype: "f16" | "f32";
  time: number;
  p50: number;
  p95: number;
  avg: number;
  size: number;
  shape?: Record<string, number>;
  useGpuTs: boolean;
  correctness?: boolean;
  gflops?: number;
  gbps?: number;
}
