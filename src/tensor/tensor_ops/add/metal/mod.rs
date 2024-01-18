pub mod metal_kernel;

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::{devices::{metal::MetalGPU, cpu::CPU}, tensor::{RandTensor, Tensor, tensor_ops::add::TryAdd}, shape::Rank1};

    #[test]
    fn test_add_metal() {
        let gpu = MetalGPU::default();

        const S: usize= 16777216;
        type T <D> = Tensor<Rank1<S>, f32, D>;

        let a: T<_> = gpu.fill_rand_range(-1.0..1.0);
        let b: T<_> = gpu.fill_rand_range(-1.0..1.0);

        let _c: T<_> = a.add(b);

        let cpu = CPU::default();

        let a: T<_> = cpu.fill_rand_range(-1.0..1.0);
        let b: T<_> = cpu.fill_rand_range(-1.0..1.0);

        let start = Instant::now();
        let _c: T<_> = a.add(b);
        let elapsed = start.elapsed();
        println!("cpu time: {:?}", elapsed);

    }
}