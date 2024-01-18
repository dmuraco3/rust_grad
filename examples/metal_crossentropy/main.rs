use std::time::Instant;

use rust_grad::{devices::{cpu::CPU, metal::MetalGPU}, tensor::{Tensor, ZerosTensor, tensor_ops::{cross_entropy::TryCrossEntropy, softmax::TrySoftmax}, RandTensor}, shape::Rank1};

fn main() {
    let cpu = CPU::default();
    let gpu = MetalGPU::default();

    let a_cpu: Tensor<Rank1<784>, f32, _> = cpu.fill_rand_range(-1.0..1.0);
    let mut a_gpu: Tensor<Rank1<784>, f32, _> = gpu.zeros();
    a_gpu.copy_from_slice(a_cpu.data.read().unwrap().as_slice());


    let c_gpu = a_gpu.softmax();
    let c_cpu = a_cpu.softmax();

    println!("{}", c_gpu.data.read().unwrap());
    println!("{}", c_cpu);
}