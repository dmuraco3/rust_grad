pub mod metal_kernel;

#[cfg(test)]
mod tests {
    use crate::{devices::{metal::MetalGPU, cpu::CPU}, tensor::{Tensor, tensor_ops::cross_entropy::TryCrossEntropy, ZerosTensor}, shape::Rank1};

    #[test]
    fn test_cross_entropy_metal() {
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

        let c_gpu = a_gpu.try_cross_entropy(b_gpu).unwrap();
        let c_cpu = a_cpu.try_cross_entropy(b_cpu).unwrap();

        println!("{} : {}", c_gpu.data.read().unwrap(), c_cpu);

    }
}