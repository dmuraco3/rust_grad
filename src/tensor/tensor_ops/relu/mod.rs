pub mod cpu;
pub mod metal;

use std::{fmt::Debug, vec};

use crate::{dtypes::Unit, shape::{Storage, Shape}, tensor::{Tensor, ZerosTensor, tape::{Tape, SplitTape, PutTape}, HasErr}};

pub trait ReLUKernel<E: Unit>: Storage<E> {
    fn forward<S: Shape>(&self, src: &Tensor<S, E, Self>, out:&mut Tensor<S,E,Self>) -> Result<(), Self::Err>;

    fn backward<S: Shape>(&self, out: &Tensor<S, E, Self>, src_grad: &mut Self::Vec, out_grad: &Self::Vec) -> Result<(), Self::Err>;
}

pub trait TryReLU<S: Shape, E: Unit, D: ReLUKernel<E>, T>: HasErr {
    type Error: Debug;

    type Output;

    fn relu(self) -> Self::Output {
        self.try_relu().unwrap()
    }

    fn try_relu(self) -> Result<Self::Output, Self::Error>;

}

impl <S, E, D, T> TryReLU<S,E,D,T> for Tensor<S, E, D, T>
where
    S: Shape + 'static,
    E: Unit,
    D: ReLUKernel<E> + ZerosTensor<E>,
    T: Tape<E, D>
{
    type Error = D::Err;
    type Output = Tensor<S, E, D, T>;

    fn try_relu(self) -> Result<Tensor<S, E, D, T>, Self::Error> {
        let mut out = self.device.try_zeros_from(&self.shape).unwrap();
        
        let (lhs, mut lhs_tape) = self.split_tape();
        
        lhs.device.forward::<S>(&lhs, &mut out)?;

        let out_d = (out.id.clone(), out.shape.num_elements());

        let t_out = out.clone();

        lhs_tape.add_backward_op(move |grads| {
            grads.try_ones_for((&lhs.device, out_d.0, out_d.1))?;
            grads.try_alloc_for((&lhs.device, lhs.id, lhs.shape.num_elements()))?;

            let out_grad = grads.get_grad_ref(&out_d.0).clone();

            let mut src_grad = grads.get_grad_mut(&lhs.id);

            lhs.device.backward(&t_out, &mut src_grad, &out_grad)?;

            Ok(())
        });
        

        Ok(out.put_tape(lhs_tape))

    }
}