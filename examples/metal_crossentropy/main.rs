use std::time::Instant;

use rust_grad::{devices::{cpu::CPU, metal::MetalGPU}, tensor::{Tensor, ZerosTensor, tensor_ops::{cross_entropy::TryCrossEntropy, softmax::TrySoftmax}}, shape::Rank1};

fn main() {
    let cpu = CPU::default();
    let gpu = MetalGPU::default();

    let a_data: [f32; 10] = [0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0];
    let b_data: [f32; 10] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0 ,0.0, 0.0, 0.0];

    let mut a_gpu: Tensor<Rank1<10>, f32, _> = gpu.zeros();
    let mut b_gpu: Tensor<Rank1<10>, f32, _> = gpu.zeros();
    a_gpu.copy_from_array(a_data);
    b_gpu.copy_from_array(b_data); 

    let a_cpu = cpu.from_array(a_data);
    let b_cpu = cpu.from_array(b_data);

    let c_gpu = a_gpu.softmax();
    let c_cpu = a_cpu.softmax();

    println!("{}", c_gpu.data.read().unwrap());
    println!("{}", c_cpu);
}