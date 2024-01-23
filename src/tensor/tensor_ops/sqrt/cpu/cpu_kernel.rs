use std::fmt::Display;

use crate::{dtypes::{Unit, FloatUnit}, devices::cpu::CPU, shape::Shape, tensor::{Tensor, tensor_ops::sqrt::SqrtKernel}};

impl <E: FloatUnit> SqrtKernel<E> for CPU {
    fn forward<S: Shape>(&self, src: &Tensor<S, E, Self>, out: &mut Tensor<S,E,Self>) -> Result<(), Self::Err> {

        let src_inner = src.data.read().unwrap();

        let mut out_inner = out.data.write().unwrap();

        for i in 0..src_inner.len() {
            out_inner[i] = src_inner[i].sqrt()
        }

        Ok(())
    }

    fn backward<S: Shape>(&self, src: &Tensor<S, E, Self>, src_grad: &mut Self::Vec, out_grad: &Self::Vec) -> Result<(), Self::Err> {
        
        todo!()
    }
    
}