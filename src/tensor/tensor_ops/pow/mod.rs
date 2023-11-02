pub mod cpu_kernel;

use std::fmt::{Debug, Display};

use crate::{dtypes::Unit, shape::{Storage, Shape}, tensor::{Tensor, ZerosTensor, tape::{Tape, SplitTape, PutTape}}};

pub trait PowKernel<E: Unit>: Storage<E> {
    fn forward<S: Shape>(&self, src: &Tensor<S, E, Self>, out:&mut Tensor<S,E,Self>, exponent: u32) -> Result<(), Self::Err>;

    fn backward<S: Shape>(&self, src: &Tensor<S, E, Self>, src_grad: &mut Self::Vec, out_grad: &Self::Vec, exponent: u32) -> Result<(), Self::Err>;
}

pub trait TryPow<S: Shape, E: Unit, D: PowKernel<E>, T> {
    type Error: Debug;

    fn try_pow(self, exponent: u32) -> Result<Tensor<S, E, D, T>, Self::Error>;
}

impl <S, E, D, T> TryPow<S,E,D, T> for Tensor<S, E, D, T>
where
    S: Shape + 'static,
    E: Unit,
    D: PowKernel<E> + ZerosTensor<E>,
    T: Tape<E, D>,
    Tensor<S, E, D>: Display
{
    type Error = D::Err;

    fn try_pow(self, exponent: u32) -> Result<Tensor<S, E, D, T>, Self::Error> {
        let mut out = self.device.try_zeros_from(&self.shape).unwrap();

        let (lhs, mut lhs_tape) = self.split_tape();

        lhs.device.forward::<S>(&lhs, &mut out, exponent)?;

        let out_d = (out.id.clone(), out.shape.num_elements());

        lhs_tape.add_backward_op(move |grads| {
            grads.try_ones_for((&lhs.device, out_d.0, out_d.1))?;

            grads.try_alloc_for((&lhs.device, lhs.id.clone(), lhs.shape.num_elements()))?;

            let out_grad = grads.get_grad_ref(&out_d.0).clone();

            let mut src_grad = grads.get_grad_mut(&lhs.id);
            

            lhs.device.backward(&lhs, &mut src_grad, &out_grad, exponent)?;

            Ok(())
        });

        Ok(out.put_tape(lhs_tape))

    }
}