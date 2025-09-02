export async function initGPU() {
  if (!("gpu" in navigator)) throw new Error("WebGPU not supported in this browser.");
  const pref = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  const adapter = pref ?? (await navigator.gpu.requestAdapter());
  if (!adapter) throw new Error("No GPU adapter found.");
  const req: GPUFeatureName[] = [];
  if (adapter.features.has("shader-f16")) req.push("shader-f16");
  if (adapter.features.has("timestamp-query")) req.push("timestamp-query");
  if ((adapter.features as any).has?.("subgroups")) req.push("subgroups" as GPUFeatureName);
  const device = await adapter.requestDevice({ requiredFeatures: req });
  let info: any = {};
  try { info = (adapter as any).info ?? (await (adapter as any).requestAdapterInfo?.()); } catch {}
  return { adapter, device, vendor: info?.vendor ?? "", features: new Set(device.features), info };
}

export function log(msg: string, prepend = true) {
  const el = document.getElementById("log");
  if (!el) { console.log(msg); return; }
  el.textContent = prepend ? (msg + "\n\n" + el.textContent) : (el.textContent + "\n\n" + msg);
}

export function createGPUBuffer(device: GPUDevice, size: number, usage: GPUBufferUsageFlags): GPUBuffer {
  return device.createBuffer({ size, usage, mappedAtCreation: false });
}

export async function runComputePipeline(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  dispatchSize: number
): Promise<void> {
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(dispatchSize);
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}
