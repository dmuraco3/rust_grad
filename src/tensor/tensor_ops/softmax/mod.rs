pub mod cpu;
pub mod metal;

use std::fmt::Debug;

use crate::{
    dtypes::Unit,
    shape::{Dim, Shape, Storage},
    tensor::{
        tape::{PutTape, SplitTape, Tape, UniqueID, Gradients},
        HasErr, Tensor, ZerosTensor,
    },
};

pub trait SoftmaxKernel<E: Unit>: Storage<E> {
    fn forward<S: Shape>(
        &self,
        src: &Tensor<S, E, Self>,
        out: &mut Tensor<S, E, Self>,
    ) -> Result<(), Self::Err>;

    fn backward<S: Shape>(
        &self,
        out_id: &UniqueID,
        grads: &mut Gradients<E, Self>
    ) -> Result<(), Self::Err>;
}

pub trait TrySoftmax<E: Unit>: HasErr {
    type Error: Debug;
    type Output;

    fn try_softmax(self) -> Result<Self::Output, Self::Error>;

    fn softmax(self) -> Self::Output {
        self.try_softmax().unwrap()
    }
}

// impl <S: Shape, E: FloatUnit, D: SoftmaxKernel<E>+ZerosTensor<E>, T: Tape<E, D> + Merge<T>> TrySoftmax<S,E,D, T> for Tensor<S,E,D, OwnedTape<E, D>> {
//     type Error = D::Err;

//     type Output = Tensor<S,E,D, OwnedTape<E, D>>;

//     fn try_softmax(self) -> Result<Self::Output, Self::Error> {
//         let mut out = self.device.try_zeros_from(&self.shape).unwrap();

//         let (lhs, lhs_tape) = self.split_tape();

//         lhs.device.forward::<S>(&lhs, &mut out)?;

//         Ok(out.put_tape(lhs_tape))
//     }
// }

impl<X: Dim, E: Unit, D: SoftmaxKernel<E> + ZerosTensor<E>, T: Tape<E, D>> TrySoftmax<E>
    for Tensor<(X,), E, D, T>
{
    type Error = D::Err;
    type Output = Tensor<(X,), E, D, T>;

    fn try_softmax(self) -> Result<Self::Output, Self::Error> {
        let mut out = self.device.try_zeros_from(&self.shape).unwrap();

        let (src, src_tape) = self.split_tape();

        src.device.forward(&src, &mut out)?;

        Ok(out.put_tape(src_tape))
    }
}

// impl <X: Dim, E: Unit, D: SoftmaxKernel<E> + ZerosTensor<E>, T: Tape<E, D>> TrySoftmax<E> for Tensor<(X, ), E, D, T> {
//     type Error = D::Err;
//     type Output = Tensor<(X, ), E, D, OwnedTape<E, D>>;

//     fn try_softmax(self) -> Result<Self::Output, Self::Error> {
//         let mut out = self.device.try_zeros_from(&self.shape).unwrap();

//         let (src, src_tape) = self.split_tape();

//         src.device.forward(&src, &mut out)?;

//         Ok(out.put_tape(src_tape))
//     }
// }
