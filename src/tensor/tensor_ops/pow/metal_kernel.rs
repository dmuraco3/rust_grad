use crate::{devices::metal::MetalGPU, dtypes::Unit, tensor::Tensor};

use super::PowKernel;

impl<E: Unit> PowKernel<E> for MetalGPU {
    fn forward<S: crate::shape::Shape>(
        &self,
        src: &Tensor<S, E, Self>,
        out: &mut Tensor<S, E, Self>,
        exponent: u32,
    ) -> Result<(), Self::Err> {
        let src_inner = src.data.read().unwrap();

        let mut out_inner = out.data.write().unwrap();

        for i in 0..src.shape.num_elements() {
            out_inner[i] = src_inner[i].pow(exponent as u16)
        }

        Ok(())
    }

    #[allow(unused_variables)]
    fn backward<S: crate::shape::Shape>(
        &self,
        src: &Tensor<S, E, Self>,
        src_grad: &mut Self::Vec,
        out_grad: &Self::Vec,
        exponent: u32,
    ) -> Result<(), Self::Err> {
        todo!()
    }
}
