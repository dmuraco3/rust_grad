pub mod cpu_kernel;

use std::fmt::Debug;

use crate::{dtypes::Unit, shape::{Storage, Shape}, tensor::{Tensor, ZerosTensor, tape::{Tape, SplitTape, PutTape}}};

pub trait ReLUKernel<E: Unit>: Storage<E> {
    fn forward<S: Shape>(&self, src: &Tensor<S, E, Self>, out:&mut Tensor<S,E,Self>) -> Result<(), Self::Err>;
}

pub trait TryReLU<S: Shape, E: Unit, D: ReLUKernel<E>, T> {
    type Error: Debug;

    fn try_relu(self) -> Result<Tensor<S, E, D, T>, Self::Error>;
}

impl <S:Shape, E:Unit, D:ReLUKernel<E>+ZerosTensor<E>, T> TryReLU<S,E,D,T> for Tensor<S, E, D, T> {
    type Error = D::Err;

    fn try_relu(self) -> Result<Tensor<S, E, D, T>, Self::Error> {
        let mut out = self.device.try_zeros_from(&self.shape).unwrap();
        
        let (lhs, lhs_tape) = self.split_tape();
        
        lhs.device.forward::<S>(&lhs, &mut out)?;

        Ok(out.put_tape(lhs_tape))

    }
}