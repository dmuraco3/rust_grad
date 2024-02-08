pub mod cpu_kernel;

use std::fmt::Debug;

use crate::{
    dtypes::Unit,
    shape::{Const, ConstDim, Dim},
    storage::Storage,
    tensor::{
        tape::{PutTape, SplitTape},
        Tensor, ZerosTensor,
    },
};

pub enum PADDING {
    VALID,
    SAME,
}

pub trait MaxPool2DKernel<E: Unit>: Storage<E> {
    fn forward<
        SrcY: Dim,
        SrcX: Dim,
        FilterY: Dim,
        FilterX: Dim,
        OutY: Dim,
        OutX: Dim,
        Stride: Dim,
    >(
        &self,
        src: &Tensor<(SrcY, SrcX), E, Self>,
        out: &mut Tensor<(OutY, OutX), E, Self>,
        filter_shape: (FilterY, FilterX),
        stride: Stride,
        padding: PADDING,
    ) -> Result<(), Self::Err>;
}

pub trait TryMaxPool2D<FilterX: Dim, FilterY: Dim, Stride: Dim, E: Unit, D: MaxPool2DKernel<E>, T> {
    type Error: Debug;

    fn try_maxpool2d_known<const X: usize, const Y: usize>(
        self,
        filter_shape: (FilterX, FilterY),
        stride: Stride,
        padding: PADDING,
    ) -> Result<Tensor<(Const<X>, Const<Y>), E, D, T>, Self::Error>;
}

impl<InpY, InpX, FilterX, FilterY, Stride, E, D, T> TryMaxPool2D<FilterX, FilterY, Stride, E, D, T>
    for Tensor<(InpY, InpX), E, D, T>
where
    InpY: ConstDim,
    InpX: ConstDim,
    FilterX: ConstDim,
    FilterY: ConstDim,
    Stride: Dim,
    E: Unit,
    D: MaxPool2DKernel<E> + ZerosTensor<E>,
{
    type Error = D::Err;

    fn try_maxpool2d_known<const X: usize, const Y: usize>(
        self,
        filter_shape: (FilterX, FilterY),
        stride: Stride,
        padding: PADDING,
    ) -> Result<Tensor<(Const<X>, Const<Y>), E, D, T>, Self::Error> {
        let mut out: Tensor<(Const<X>, Const<Y>), E, D> = self.device.zeros();

        let (lhs, lhs_tape) = self.split_tape();

        lhs.device
            .forward(&lhs, &mut out, filter_shape, stride, padding)?;

        Ok(out.put_tape(lhs_tape))
    }
}
