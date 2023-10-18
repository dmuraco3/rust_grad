use crate::{dtypes::Unit, devices::cpu::CPU, shape::{Shape, Rank0}, tensor::Tensor};

use super::SumKernel;

impl <E: Unit> SumKernel<E> for CPU {
    fn forward<S: Shape>(&self, src: &Tensor<S, E, Self>, out: &mut Tensor<Rank0,E,Self>) -> Result<(), Self::Err> {

        let src_inner = src.data.read().unwrap();
        let mut out_inner = out.data.write().unwrap();

        // src_inner.iter().zip(out_inner.iter_mut()).for_each(|(lh, rh)| *rh=lh.pow(exponent));

        out_inner[0] = src_inner.iter().fold(E::ZERO, |acc, src_ele| acc + *src_ele );

        Ok(())
    }

    fn backward<S: Shape>(
        &self,
        src: &Tensor<S, E, Self>,
        src_grad: &mut Self::Vec,
        out_grad: &Self::Vec,
    ) -> Result<(), Self::Err> {
        debug_assert_eq!(out_grad.len(), 1);
        let out = out_grad[0];
        let one = E::ONE;
        for i in 0..src.shape.num_elements() {
            src_grad[i] = out * one;
        }

        Ok(())
    }
}