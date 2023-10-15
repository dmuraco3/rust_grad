pub mod cpu_kernel;

use std::fmt::Debug;

use crate::{dtypes::Unit, shape::{Storage, Shape}, tensor::{Tensor, ZerosTensor, tape::{Tape, SplitTape, PutTape}}};

pub trait PowKernel<E: Unit>: Storage<E> {
    fn forward<S: Shape>(&self, src: &Tensor<S, E, Self>, out:&mut Tensor<S,E,Self>, exponent: u32) -> Result<(), Self::Err>;
}

pub trait TryPow<S: Shape, E: Unit, D: PowKernel<E>, T> {
    type Error: Debug;

    fn try_pow(self, exponent: u32) -> Result<Tensor<S, E, D, T>, Self::Error>;
}

impl <S:Shape, E:Unit, D:PowKernel<E> + ZerosTensor<E>, T> TryPow<S,E,D, T> for Tensor<S, E, D, T> {
    type Error = D::Err;

    fn try_pow(self, exponent: u32) -> Result<Tensor<S, E, D, T>, Self::Error> {
        let mut out = self.device.try_zeros_from(&self.shape).unwrap();

        let (lhs, lhs_tape) = self.split_tape();

        lhs.device.forward::<S>(&lhs, &mut out, exponent)?;

        Ok(out.put_tape(lhs_tape))

    }
}