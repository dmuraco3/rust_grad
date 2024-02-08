pub mod metal_kernel;

#[cfg(test)]
mod tests {
    use crate::{devices::metal::MetalGPU, shape::Const, tensor::tensor_ops::matmul::TryMatMul};

    #[test]
    fn test_matmul() {
        let device = MetalGPU::default();

        let a: crate::tensor::Tensor<(Const<4>, Const<4>), f32, MetalGPU> = device.from_2d_array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);

        let b = device.from_array([1.0, 2.0, 3.0, 4.0]);

        let _c = a.matmul(b.clone());
    }
}
