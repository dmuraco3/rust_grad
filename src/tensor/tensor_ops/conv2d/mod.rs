pub mod cpu_kernel;

use std::{fmt::Debug, error};

use crate::{dtypes::Unit, shape::{Storage, Dim, Shape, ConstShape, ConstDim, Rank2, HasShape, Const}, tensor::{Tensor, ZerosTensor}};

#[derive(Debug, Copy, Clone)]
pub struct Conv2DOp {
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub in_shape: (usize, usize),
    pub filter_shape: (usize, usize),
    pub out_shape: (usize, usize),
}

// pub trait Conv2DKernel<E: Unit>: Storage<E> {
//     fn alloc<S: Shape>(&self, s: S) -> Result<Tensor<S, E, Self>, Self::Err>;

//     fn forward<InpShape: ConstShape, FilterShape: ConstShape, OutShape: Shape>(
//         &self,
//         inp: &Tensor<InpShape, E, Self>,
//         filter: &Tensor<FilterShape, E, Self>,
//         out: &mut Tensor<OutShape, E, Self>,
//         op: &Conv2DOp
//     ) -> Result<(), Self::Err>;

// }

// pub trait TryConv2D<KernelX: Dim, KernelY: Dim, Stride: Dim, Padding: Dim, Dilation: Dim, Groups: Dim, E: Unit, D: Storage<E>>: Sized {
//     type Output;
//     type Error: Debug;

//     fn conv2d(
//         self,
//         weights: &Tensor<(KernelX, KernelY), E, D>,
//         stride: Stride,
//         padding: Padding,
//         dilation: Dilation,
//         groups: Groups,
//     ) -> Self::Output {
//         self.try_conv2d(weights, stride, padding, dilation, groups).unwrap()
//     }

//     fn try_conv2d(
//         self,
//         weights: &Tensor<(KernelX, KernelY), E, D>,
//         stride: Stride,
//         padding: Padding,
//         dilation: Dilation,
//         groups: Groups,
//     ) -> Result<Self::Output, Self::Error>;

//     /// Can't get [Const] output shape of conv2d without generic_const_expr which requires nightly
//     /// 
//     /// User can calculate the output shape of conv2d with the following formula
//     /// 
//     /// output_dim = (input_dim - kernel_dim + 2 * padding) / stride + 1
//     /// 
//     fn try_conv2d_known<const X: usize, const Y: usize>(
//         self,
//         weights: &Tensor<(KernelX, KernelY), E, D>,
//         stride: Stride,
//         padding: Padding,
//         dilation: Dilation,
//         groups: Groups,
//     ) -> Result<Tensor<(Const<X>, Const<Y>), E, D>, Self::Error>;

// }

// impl <InpX, InpY, KernelX, KernelY, Stride, Padding, Dilation, Groups, E, D, T>
// TryConv2D<KernelX, KernelY, Stride, Padding, Dilation, Groups, E, D> 
// for Tensor<(InpX, InpY), E, D, T>
// where
//     InpX: ConstDim,
//     InpY: ConstDim,
//     KernelX: ConstDim,
//     KernelY: ConstDim,
//     Stride: Dim,
//     Padding: Dim,
//     Dilation: Dim, 
//     Groups: Dim,
//     E: Unit,
//     D: Conv2DKernel<E> + ZerosTensor<E, T>,
// {

//     type Output = Tensor<
//         <(InpX, InpY, KernelX, KernelY) as OutShapeConv2D<KernelX, KernelY, Stride, Padding, Dilation, InpX, InpY>>::OutShape,
//         E,
//         D
//     >;



//     type Error = D::Err;

//     fn try_conv2d(
//         self,
//         weights: &Tensor<(KernelX, KernelY), E, D>,
//         stride: Stride,
//         padding: Padding,
//         dilation: Dilation,
//         groups: Groups,
//     ) -> Result<Self::Output, Self::Error> {

//         // todo!();
//         let out_shape = (self.shape.0.size(), self.shape.1.size(), weights.shape.0.size(), weights.shape.1.size()).return_conv2d_shape(stride, padding, dilation);
//         let x = self.device.try_zeros_from(&out_shape).unwrap();
//         Ok(x)
//     }

//     fn try_conv2d_known<const X: usize, const Y: usize> (
//         self,
//         weights: &Tensor<(KernelX, KernelY), E, D>,
//         stride: Stride,
//         padding: Padding,
//         dilation: Dilation,
//         groups: Groups,
//     ) -> Result<Tensor<(Const<X>, Const<Y>), E, D>, Self::Error> 
//     {
//         let mut out: Tensor<(Const<X>, Const<Y>), E, D> = self.device.zeros();

//         let op = Conv2DOp {
//             stride: stride.size(),
//             padding: padding.size(),
//             dilation: dilation.size(),
//             in_shape: (self.shape.0.size(), self.shape.1.size()),
//             filter_shape: (weights.shape.0.size(), weights.shape.1.size()),
//             out_shape: (out.shape.0.size(), out.shape.1.size()),
//         };

//         self.device.forward::<_,_,(Const<X>, Const<Y>)>(
//             &self,
//             weights,
//             &mut out,
//             &op,
//         )?;

//         return Ok(out);
//     }

    
// }


// pub trait OutShapeConv2D<
// KernelX,
// KernelY,
// Stride,
// Padding,
// Dilation,
// InpX,
// InpY,
// > {
//     type OutShape;

//     fn return_conv2d_shape(
//         self,
//         stride: Stride,
//         padding: Padding,
//         dilation: Dilation,
//     ) -> Self::OutShape;
// }

// #[cfg(feature = "nightly")]
// impl <
//     const OUTCHAN: usize,
//     const KERNELX: usize,
//     const KERNELY: usize,
//     const STRIDE: usize,
//     const PADDING: usize,
//     const DILATION: usize,
//     const INPX: usize,
//     const INPY: usize,
// > OutShapeConv2D <
//     OUTCHAN,
//     KERNELX,
//     KERNELY,
//     STRIDE,
//     PADDING,
//     DILATION,
//     INPX,
//     INPY
// >  for (Const<INPX>, Const<INPY>, Const<KERNELX>, Const<KERNELY>) 
// where
//     Const<{(INPX + 2 * PADDING - DILATION * (KERNELX - 1) - 1) / STRIDE + 1}>: Sized,
// {
//     type OutShape = Const<{(INPX + 2 * PADDING - DILATION * (KERNELX - 1) - 1) / STRIDE + 1}>;
// }

// impl <
//     KERNELX: Dim,
//     KERNELY: Dim,
//     STRIDE: Dim,
//     PADDING: Dim,
//     DILATION: Dim,
//     INPX: Dim,
//     INPY: Dim
// > OutShapeConv2D <
//     KERNELX,
//     KERNELY,
//     STRIDE,
//     PADDING,
//     DILATION,
//     INPX,
//     INPY
// >  for (INPX, INPY, KERNELX, KERNELY)
// {
//     type OutShape = (usize, usize);

//     fn return_conv2d_shape (
//         self,
//         stride: STRIDE,
//         padding: PADDING,
//         dilation: DILATION,
//     ) -> Self::OutShape {
//         let (inp_x, inp_y, kernel_x, kernel_y) = self;
//         let x = (inp_x.size() + 2 * padding.size() - 1).checked_sub(dilation.size() * (kernel_x.size() - 1)).unwrap().checked_div(stride.size()).unwrap() + 1;
//         let y = (inp_y.size() + 2 * padding.size() - 1).checked_sub(dilation.size() * (kernel_y.size() - 1)).unwrap().checked_div(stride.size()).unwrap() + 1;

//         (x, y)
//     }

    
// }
// pub trait KnownConv2DShape: Shape + ConstShape {
// }

// impl <const X:usize, const Y:usize> KnownConv2DShape for (Const<X>, Const<Y>) {
// }

// // impl KnownConv2DShape for (usize, usize, usize) {}