use crate::{dtypes::Unit, devices::cpu::CPU, shape::Shape, tensor::Tensor};

use super::PowKernel;

impl <E: Unit> PowKernel<E> for CPU {
    fn forward<S: Shape>(&self, src: &Tensor<S, E, Self>, out: &mut Tensor<S,E,Self>, exponent: u32) -> Result<(), Self::Err> {

        let src_inner = src.data.read().unwrap();
        let mut out_inner = out.data.write().unwrap();

        // src_inner.iter().zip(out_inner.iter_mut()).for_each(|(lh, rh)| *rh=lh.pow(exponent));

        for i in 0..src_inner.len() {
            out_inner[i] = src_inner[i].pow(exponent)
        }

        Ok(())
    }
}