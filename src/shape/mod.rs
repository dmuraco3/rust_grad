use std::{fmt::{Debug, Display}, ops::IndexMut};

use crate::tensor::HasErr;

pub trait Dim: 'static + Copy + Clone + Debug + PartialEq {
    fn size(&self)->usize;
}

pub trait ConstDim: Default + Dim {
    const SIZE: usize;
}

impl Dim for usize {
    fn size(&self)->usize {
        *self
    }
}

#[derive(Clone, Copy, Default, PartialEq)]
pub struct Const<const X:usize>;

impl <const X:usize> Display for Const<X> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", X)
    }
}
impl <const X:usize> Debug for Const<X> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", X)
    }
}

impl<const X:usize> Dim for Const<X> {
    fn size(&self)->usize {
        X
    }
}

impl<const X: usize> ConstDim for Const<X> {
    const SIZE: usize = X;
}

pub trait Shape: Debug + Copy {

    type Concrete: IntoIterator<Item=usize>;

    fn concrete(&self) -> Self::Concrete;
    
    fn num_elements(&self) -> usize {
        self.concrete().into_iter().product()
    }
}

pub type Rank0 = ();

pub type Rank1<const X:usize> = (Const<X>,);

pub type Rank2<const X:usize, const Y:usize> = (Const<X>, Const<Y>);

pub type Rank3<const X: usize, const Y: usize, const Z: usize> = (Const<X>, Const<Y>, Const<Z>);

impl Shape for () {
    type Concrete = [usize;0];

    fn concrete(&self) -> Self::Concrete {
        []
    }
}

impl <X: Dim> Shape for (X,) {
    type Concrete=[usize;1];

    fn concrete(&self) -> Self::Concrete {
        return [self.0.size()];
    }

    fn num_elements(&self) -> usize {
        self.concrete().into_iter().sum()
    }
}

impl <X:Dim,Y:Dim> Shape for (X,Y) {
    type Concrete=[usize;2];

    fn concrete(&self) -> Self::Concrete {
        return [self.0.size(),self.1.size()];
    }
}

impl <X: Dim, Y: Dim, Z: Dim> Shape for (X, Y, Z) {
    type Concrete = [usize;3];

    fn concrete(&self) -> Self::Concrete {
        return [self.0.size(), self.1.size(), self.2.size()];
    }
}

impl ConstShape for () {
    const NumElements: usize = 1;
}

impl <X: ConstDim> ConstShape for (X,) {
    const NumElements: usize = X::SIZE;
}

impl <X:ConstDim, Y:ConstDim> ConstShape for (X, Y) {
    const NumElements: usize = X::SIZE * Y::SIZE;
}

impl <X: ConstDim, Y: ConstDim, Z: ConstDim> ConstShape for (X, Y, Z) {
    const NumElements: usize = X::SIZE * Y::SIZE * Z::SIZE;
}

pub trait Storage<E>: 'static + std::fmt::Debug + Default + Clone + HasErr {
    type Vec: 'static + Debug + Clone + IndexMut<usize, Output = E> + IntoIterator<Item = E>;

    fn try_alloc_len(&self, len: usize) -> Result<Self::Vec, Self::Err>;

    fn try_alloc_ones(&self, len: usize) -> Result<Self::Vec, Self::Err>;

    fn num_el(&self, st: Self::Vec) -> usize;
}

pub trait HasShape {
    type WithShape<New: Shape>: HasShape<Shape=New>;
    type Shape: Shape;
    fn shape(&self) -> &Self::Shape;
}

pub trait ConstShape: Default + Shape {
    const NumElements: usize;
}

impl<S: Shape> HasShape for S {
    type WithShape<New: Shape> = New;
    type Shape = Self;
    fn shape(&self) -> &Self::Shape {
        self
    }
}