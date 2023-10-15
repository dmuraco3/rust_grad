use crate::{dtypes::{Unit, FloatUnit}, devices::cpu::CPU, shape::Shape, tensor::Tensor};

use super::SCCEKernel;

impl <E: FloatUnit> SCCEKernel<E> for CPU {
    fn forward<S: Shape, OutShape: Shape>(src: &Tensor<S, E, Self>, actual: &Tensor<S, E, Self>, out: &mut Tensor<OutShape, E, Self>) -> Result<(), Self::Err> {


        let loss = src.data.read().unwrap().iter().zip(actual.data.read().unwrap().iter()).fold(E::ZERO, |acc, (&p, &t): (&E, &E)| acc + t * p.log());
        
        out.data.write().unwrap()[0] = loss;

        Ok(())
    }
}