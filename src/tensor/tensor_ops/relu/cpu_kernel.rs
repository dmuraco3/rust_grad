use crate::{dtypes::Unit, devices::cpu::CPU, shape::Shape, tensor::Tensor};

use super::ReLUKernel;

impl <E: Unit> ReLUKernel<E> for CPU {
    fn forward<S: Shape>(&self, src: &Tensor<S, E, Self>, out: &mut Tensor<S,E,Self>) -> Result<(), Self::Err> {

        let src_inner = src.data.read().unwrap();
        let mut out_inner = out.data.write().unwrap();

        src_inner.iter().zip(out_inner.iter_mut()).for_each(|(lh, rh)| *rh=lh.max(*rh));

        Ok(())
    }
}