use std::{fmt::Debug, process::Output};

use crate::{shape::{Shape, Storage, Dim, ConstDim, Const}, dtypes::{FloatUnit, Unit}, tensor::{Tensor, ZerosTensor, tape::Tape}};

pub mod cpu_kernel;

pub trait SCCEKernel<E: Unit>: Storage<E> {
    fn forward<S: Shape, OutShape: Shape>(src: &Tensor<S, E, Self>, actual: &Tensor<S, E, Self>, out: &mut Tensor<OutShape, E, Self>) -> Result<(), Self::Err>;
}

pub trait TrySparseCategoricalCrossentropy<S: Shape, E: Unit, D: SCCEKernel<E>> {
    type Error: Debug;

    type Output;

    fn try_sparse_categorical_crossentropy(self, actual: Tensor<S, E, D>) -> Result<Self::Output, Self::Error>;
}

impl <X: ConstDim, E: Unit, D: SCCEKernel<E> + ZerosTensor<E>> TrySparseCategoricalCrossentropy<(X,), E, D> for Tensor<(X,), E, D> {
    type Error = D::Err;

    type Output = Tensor<(Const<1>, ), E, D>;

    fn try_sparse_categorical_crossentropy(self, actual: Tensor<(X,), E, D>) -> Result<Self::Output, Self::Error> {
        let mut out = self.device.zeros::<(Const<1>, )>();

        D::forward(&self, &actual, &mut out)?;

        Ok(out)
    }
}