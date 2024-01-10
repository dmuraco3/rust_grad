use std::fmt::Debug;

use crate::{dtypes::Unit, shape::{Storage, Shape}, tensor::{Tensor, tape::{Gradients, UniqueID}, HasErr}};

pub trait ReshapeKernel <E: Unit>: Storage<E> {
    fn forward<OS: Shape, NS: Shape>(&self, lhs: &Tensor<OS, E, Self>, out: &mut Tensor<NS, E, Self>) -> Result<(), Self::Err>;

    fn backward<OS: Shape, NS: Shape>(&self, grads: &mut Gradients<E, Self>, lhs_id: &UniqueID, out_id: &UniqueID) -> Result<(), Self::Err>;
}

pub trait TryReshape<S: Shape, E: Unit, D: ReshapeKernel<E>>: HasErr {
    type Error: Debug;
    type Output;

    fn try_reshape(self, shape: S) -> Result<Self::Output, Self::Error>;

    fn reshape(self, shape: S) -> Self::Output {
        self.try_reshape(shape).unwrap()
    }
}

impl <OS, NS, E, D> TryReshape <NS, E, D> for Tensor<OS, E, D>
where
    OS: Shape,
    NS: Shape,
    E: Unit, 
    D: ReshapeKernel<E>

{
    type Error = D::Err;

    type Output = Tensor<NS, E, D>;

    fn try_reshape(self, shape: NS) -> Result<Self::Output, Self::Error> {
        todo!()
    }
}