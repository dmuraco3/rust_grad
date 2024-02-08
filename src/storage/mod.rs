use std::fmt::Debug;
use std::ops::IndexMut;

use crate::{shape::TensorInnerLength, tensor::HasErr};

pub trait Storage<E>: 'static + std::fmt::Debug + Default + Clone + HasErr {
    type Vec: 'static
        + Debug
        + Clone
        + IndexMut<usize, Output = E>
        + IntoIterator<Item = E>
        + TensorInnerLength<E>;

    fn try_alloc_len(&self, len: usize) -> Result<Self::Vec, Self::Err>;

    fn try_alloc_ones(&self, len: usize) -> Result<Self::Vec, Self::Err>;

    fn num_el(&self, st: Self::Vec) -> usize;
}
