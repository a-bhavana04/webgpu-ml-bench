import { initGPU, log } from "./util";
import sanityWGSL from "./kernels/sanity.wgsl?raw";
import { benchMatmulF32, benchMatmulF16 } from "./bench/matmul";
import { autotuneMatmul } from "./bench/matmul_autotune";
import { benchQKScore } from "./bench/qk_score";
import { benchSoftmax } from "./bench/softmax";
import { benchLayerNorm } from "./bench/layernorm";
import { benchGELU } from "./bench/gelu";
import { benchMiniLMEmbeddings } from "./bench/e2e_minilm";
import { benchTensorFlow } from "./bench/tensorflow";
import type { BenchResult } from "./bench/types";

function formatShape(r: Pick<BenchResult, "shape" | "size">): string {
  const s = r.shape;
  if (!s) return String(r.size);
  if ("M" in s && "N" in s && "K" in s) return `${s.M}×${s.N}×${s.K}`;
  if ("B" in s && "N" in s) return `${s.B}×${s.N}`;
  if ("rows" in s && "cols" in s) return `${(s as any).rows}×${(s as any).cols}`;
  if ("size" in s) return String((s as any).size);
  const vals = Object.values(s);
  return vals.length ? vals.join("×") : String(r.size);
}

function perfString(r: BenchResult): string {
  const v = r.gflops ?? r.gbps;
  return v != null ? v.toFixed(2) : "-";
}

function addRow(r: {
  op: string;
  shape: { M: number; N: number; K: number };
  tiles: { tm: number; tn: number; tk: number };
  dtype: string;
  vendor: string;
  useGpuTs: boolean;
  p50: number;
  p95: number;
  gflops: number;
  first: number;
  expect: number;
}) {
  const tb = document.querySelector<HTMLTableSectionElement>("#results tbody")!;
  const tr = document.createElement("tr");
  tr.innerHTML = [
    r.op,
    `${r.shape.M}×${r.shape.N}×${r.shape.K}`,
    `${r.tiles.tm}/${r.tiles.tn}/${r.tiles.tk}`,
    r.dtype,
    r.vendor,
    String(r.useGpuTs),
    r.p50.toFixed(3),
    r.p95.toFixed(3),
    r.gflops.toFixed(2),
    `${r.first.toFixed(2)}~${r.expect}`,
  ]
    .map((x) => `<td>${x}</td>`)
    .join("");
  tb.appendChild(tr);
}

function addRowGeneric(r: BenchResult) {
  const tb = document.querySelector<HTMLTableSectionElement>("#results tbody")!;
  const tr = document.createElement("tr");
  tr.innerHTML = [
    r.op,
    formatShape(r),
    "-",
    r.dtype,
    r.vendor,
    String(r.useGpuTs),
    r.p50.toFixed(3),
    r.p95.toFixed(3),
    perfString(r),
    r.avg.toFixed(3),
  ]
    .map((x) => `<td>${x}</td>`)
    .join("");
  tb.appendChild(tr);
}

window.addEventListener("DOMContentLoaded", () => {
  const btnDetect = document.getElementById("btn-detect") as HTMLButtonElement | null;
  const btnSanity = document.getElementById("btn-sanity") as HTMLButtonElement | null;
  const btnF32 = document.getElementById("btn-matmul-f32") as HTMLButtonElement | null;
  const btnF16 = document.getElementById("btn-matmul-f16") as HTMLButtonElement | null;
  const btnAuto = document.getElementById("btn-matmul-auto") as HTMLButtonElement | null;
  const btnQKScore = document.getElementById("btn-qk-score") as HTMLButtonElement | null;
  const btnSoftmax = document.getElementById("btn-softmax") as HTMLButtonElement | null;
  const btnLayernorm = document.getElementById("btn-layernorm") as HTMLButtonElement | null;
  const btnGELU = document.getElementById("btn-gelu") as HTMLButtonElement | null;
  const btnE2E = document.getElementById("btn-e2e-minilm") as HTMLButtonElement | null;
  const btnTensorFlow = document.getElementById("btn-tensorflow") as HTMLButtonElement | null;

  const iM = document.getElementById("m") as HTMLInputElement | null;
  const iN = document.getElementById("n") as HTMLInputElement | null;
  const iK = document.getElementById("k") as HTMLInputElement | null;
  const iR = document.getElementById("repeats") as HTMLInputElement | null;
  const iTM = document.getElementById("tm") as HTMLInputElement | null;
  const iTN = document.getElementById("tn") as HTMLInputElement | null;
  const iTK = document.getElementById("tk") as HTMLInputElement | null;
  const iDT = document.getElementById("dtype") as HTMLSelectElement | null;
  const iTexts = document.getElementById("texts") as HTMLTextAreaElement | null;
  const iRepeats2 = document.getElementById("repeats-e2e") as HTMLInputElement | null;

  if (btnDetect)
    btnDetect.onclick = async () => {
      try {
        const { device, vendor, features, info } = await initGPU();
        log(
          [
            "WebGPU available",
            `Vendor: ${vendor || info?.vendor || "(unknown)"}`,
            `Architecture: ${info?.architecture ?? "(n/a)"}`,
            `Device: ${info?.device ?? "(n/a)"}`,
            `Features: ${[...features].join(", ") || "(none)"}`,
            `Limits: maxComputeWorkgroupSizeX=${device.limits.maxComputeWorkgroupSizeX}`,
          ].join("\n"),
          true
        );
      } catch (e: any) {
        log("Error: " + e.message, true);
      }
    };

  if (btnSanity)
    btnSanity.onclick = async () => {
      try {
        const { device } = await initGPU();
        const out = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const read = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const module = device.createShaderModule({ code: sanityWGSL });
        const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
        const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: out } }] });
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(1);
        pass.end();
        enc.copyBufferToBuffer(out, 0, read, 0, 4);
        device.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const v = new Uint32Array(read.getMappedRange())[0];
        read.unmap();
        log(`Sanity compute wrote value: ${v} (expected 42)`, true);
      } catch (e: any) {
        log("Error: " + e.message, true);
      }
    };

  const readInputs = () => ({
    M: parseInt(iM!.value, 10),
    N: parseInt(iN!.value, 10),
    K: parseInt(iK!.value, 10),
    R: parseInt(iR!.value, 10),
    tm: parseInt(iTM!.value, 10),
    tn: parseInt(iTN!.value, 10),
    tk: parseInt(iTK!.value, 10),
    dtype: iDT!.value as "f32" | "f16",
  });

  if (btnF32)
    btnF32.onclick = async () => {
      const v = readInputs();
      const r = await benchMatmulF32(v.M, v.N, v.K, { tm: v.tm, tn: v.tn, tk: v.tk, repeats: v.R });
      addRow(r);
    };

  if (btnF16)
    btnF16.onclick = async () => {
      const v = readInputs();
      const r = await benchMatmulF16(v.M, v.N, v.K, { tm: v.tm, tn: v.tn, tk: v.tk, repeats: v.R });
      addRow(r);
    };

  if (btnAuto)
    btnAuto.onclick = async () => {
      const v = readInputs();
      const { best, results } = await autotuneMatmul(v.M, v.N, v.K, v.R, v.dtype);
      results.forEach(addRow);
      log(`best tiles ${best.tiles.tm}/${best.tiles.tn}/${best.tiles.tk} p50=${best.p50.toFixed(3)}ms`, true);
    };

  if (btnQKScore)
  if (btnQKScore)
  btnQKScore.onclick = async () => {
    const size = parseInt((document.getElementById("size") as HTMLInputElement).value, 10);
    const dtype = (document.getElementById("dtype") as HTMLSelectElement).value as "f32" | "f16";
    const r = await benchQKScore(size, dtype);
    addRowGeneric(r);
  };

if (btnSoftmax)
  btnSoftmax.onclick = async () => {
    const size = parseInt((document.getElementById("size") as HTMLInputElement).value, 10);
    const dtype = (document.getElementById("dtype") as HTMLSelectElement).value as "f32" | "f16";
    const repeats = (document.getElementById("repeats") as HTMLInputElement)?.value
      ? parseInt((document.getElementById("repeats") as HTMLInputElement).value, 10)
      : 10;
    const B = size;
    const N = size;
    const r = await benchSoftmax(B, N, dtype, repeats);
    addRowGeneric(r);
  };

if (btnLayernorm)
  btnLayernorm.onclick = async () => {
    const size = parseInt((document.getElementById("size") as HTMLInputElement).value, 10);
    const dtype = (document.getElementById("dtype") as HTMLSelectElement).value as "f32" | "f16";
    const repeats = (document.getElementById("repeats") as HTMLInputElement)?.value
      ? parseInt((document.getElementById("repeats") as HTMLInputElement).value, 10)
      : 10;
    const B = size;
    const N = size;
    const useAffine = true;
    const eps = 1e-5;
    const r = await benchLayerNorm(B, N, useAffine, eps, dtype, repeats);
    addRowGeneric(r);
  };

if (btnGELU)
  btnGELU.onclick = async () => {
    const size = parseInt((document.getElementById("size") as HTMLInputElement).value, 10);
    const dtype = (document.getElementById("dtype") as HTMLSelectElement).value as "f32" | "f16";
    const r = await benchGELU(size, dtype);
    addRowGeneric(r);
  };

  if (btnE2E)
  btnE2E.onclick = async () => {
    const raw = (iTexts?.value ?? "").trim();
    const dtype = (document.getElementById("dtype") as HTMLSelectElement).value as "f32" | "f16";
    const repeats = iRepeats2?.value ? parseInt(iRepeats2.value, 10) : 5;
    const texts = raw ? raw.split(/\n+/).map(s => s.trim()).filter(Boolean) : ["hello world", "webgpu ml bench", "transformers.js minilm embeddings"];
    const r = await benchMiniLMEmbeddings(texts, dtype, repeats);
    addRowGeneric(r);
  };

  if (btnTensorFlow)
  btnTensorFlow.onclick = async () => {
    const modelUrl = "https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/model.json";
    const inputShape = [1, 224, 224, 3]; 
    const r = await benchTensorFlow(modelUrl, inputShape, 5);
    addRowGeneric(r);
  };
});
