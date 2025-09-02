struct Out {
  data: array<u32>,
};

@group(0) @binding(0) var<storage, read_write> OUT : Out;

@compute @workgroup_size(1)
fn main() {
  OUT.data[0] = 42u;
}
