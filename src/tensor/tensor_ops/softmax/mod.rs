pub mod cpu_kernel;

use std::fmt::Debug;

use crate::{dtypes::{Unit, FloatUnit}, shape::{Storage, Shape}, tensor::{Tensor, ZerosTensor, tape::{Tape, SplitTape, PutTape}}};

pub trait SoftmaxKernel<E: FloatUnit>: Storage<E> {
    fn forward<S: Shape>(&self, src: &Tensor<S, E, Self>, out: &mut Tensor<S, E, Self>) -> Result<(), Self::Err>;
}

pub trait TrySoftmax<S: Shape, E: FloatUnit, D: SoftmaxKernel<E>, T> {
    type error: Debug;
    
    fn try_softmax(self) -> Result<Tensor<S,E,D, T>, Self::error>;
}

impl <S: Shape, E: FloatUnit, D: SoftmaxKernel<E>+ZerosTensor<E>, T> TrySoftmax<S,E,D, T> for Tensor<S,E,D, T> {
    type error = D::Err;

    fn try_softmax(self) -> Result<Tensor<S,E,D, T>, Self::error> {
        let mut out = self.device.try_zeros_from(&self.shape).unwrap();

        let (lhs, lhs_tape) = self.split_tape();

        lhs.device.forward::<S>(&lhs, &mut out)?;

        Ok(out.put_tape(lhs_tape))
    }
}