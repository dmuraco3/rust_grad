use crate::{devices::cpu::CPU, dtypes::Unit, shape::Shape, tensor::Tensor};

use super::PowKernel;

impl<E: Unit> PowKernel<E> for CPU {
    fn forward<S: Shape>(
        &self,
        src: &Tensor<S, E, Self>,
        out: &mut Tensor<S, E, Self>,
        exponent: u32,
    ) -> Result<(), Self::Err> {
        let src_inner = src.data.read().unwrap();

        let mut out_inner = out.data.write().unwrap();

        for i in 0..src_inner.len() {
            out_inner[i] = src_inner[i].pow(exponent as u16)
        }

        Ok(())
    }

    fn backward<S: Shape>(
        &self,
        src: &Tensor<S, E, Self>,
        src_grad: &mut Self::Vec,
        out_grad: &Self::Vec,
        exponent: u32,
    ) -> Result<(), Self::Err> {
        let src_data = src.data.read().unwrap();

        src_grad
            .iter_mut()
            .zip(out_grad.iter())
            .for_each(|(src, out)| {
                *src = *out * E::from_u32(exponent);
            });

        for i in 0..src_data.len() {
            src_grad[i] = out_grad[i] * src_data[i] * E::from_u32(exponent);
        }

        Ok(())
    }
}
