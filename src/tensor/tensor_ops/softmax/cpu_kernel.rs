use crate::{dtypes::{Unit, FloatUnit}, devices::cpu::CPU, shape::Shape, tensor::Tensor};

use super::SoftmaxKernel;

impl <E: FloatUnit> SoftmaxKernel<E> for CPU {
    fn forward<S: Shape>(&self, src: &Tensor<S, E, Self>, out: &mut Tensor<S, E, Self>) -> Result<(), Self::Err> {
        let src_inner = src.data.read().unwrap();
        let mut out_inner = out.data.write().unwrap();

        let divisor = src_inner.iter().fold(E::ZERO, |acc, &x| acc + x.exp());

        src_inner.iter().zip(out_inner.iter_mut()).for_each(|(src, out)| {
            *out = src.exp() / divisor;
        });

        Ok(())
    }
}