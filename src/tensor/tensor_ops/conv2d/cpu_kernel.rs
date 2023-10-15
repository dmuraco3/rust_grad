// use std::{borrow::BorrowMut, ops::{IndexMut, Index}, sync::RwLockWriteGuard};

// use crate::{dtypes::Unit, devices::cpu::CPU, shape::{Shape, ConstShape, HasShape, ConstDim}, tensor::{ZerosTensor, Tensor, tape::Tape}};

// use super::{Conv2DKernel, KnownConv2DShape, Conv2DOp};


// impl<E: Unit, T: Tape<E, Self>> Conv2DKernel<E> for CPU
// where
//     Self: ZerosTensor<E, T>
// {
//     fn alloc<S: Shape>(&self, s: S) -> Result<Tensor<S, E, Self, T>, Self::Err> {
//         self.try_zeros_from(&s)
//     }

//     fn forward<InShape: ConstShape, FilterShape: ConstShape, OutShape: Shape>(
//         &self,
//         inp: &Tensor<InShape, E, Self>,
//         filter: &Tensor<FilterShape, E, Self>,
//         out: &mut Tensor<OutShape, E, Self>,
//         op: &Conv2DOp,
//     ) -> Result<(), Self::Err>
//     {
//         let Conv2DOp {stride, padding, dilation, out_shape, in_shape, filter_shape } = op;

//         let mut out_inner = out.data.write().unwrap();
//         let inp_inner = inp.data.read().unwrap();
//         let filter_inner = filter.data.read().unwrap();

//         for out_y in 0..out_shape.0 {
//             for out_x in 0..out_shape.1 {
//                 for filter_y in 0..filter_shape.0 {
//                     for filter_x in 0..filter_shape.1 {
//                         let l: &E = inp_inner.index((in_shape.1 * (filter_y + out_y)) + (out_x + filter_x));
//                         // in_patch_index = out_shape.X * (out_ind_y + filter_y) + (out_x + filter_x) 
//                         *out_inner.index_mut(out_shape.1 * out_y + out_x) += *l * *filter_inner.index(filter_shape.1*filter_y + filter_x);

//                     };
//                 };
//             }
//         }

//         return Ok(());
//     }
// }