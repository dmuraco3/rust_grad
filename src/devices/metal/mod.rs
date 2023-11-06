use std::{fmt::Display, sync::{Arc, RwLock}, marker::PhantomData, clone, ops::{IndexMut, Index}, ffi::c_void, mem::size_of};

use metal::{DeviceRef, Device, MTLResourceOptions};

use crate::{shape::{Shape, Storage, HasShape}, dtypes::Unit, tensor::{HasErr, ZerosTensor, Tensor, tape::{unique_id, NoneTape}}};

#[derive(Debug, Clone)]
pub struct MetalGPU {
    pub(crate) device: Device,
}

impl Default for MetalGPU {
    fn default() -> Self {

        let device = Device::system_default().expect("No metal device found");

        Self {
            device
        }
    }
}


/// Vector backed by GPU memory
#[derive(Clone, Debug)]
pub struct MetalVec<E> {
    buf: metal::Buffer,
    len: usize,

    _marker: PhantomData<E>
}

impl <E> Index<usize> for MetalVec<E> {
    type Output = E;

    fn index(&self, index: usize) -> &Self::Output {
        let ptr = self.buf.contents() as *const c_void;

        let ptr = ptr.cast::<E>();

        let data = unsafe {
            ptr.as_ref().unwrap()
        };
        
        data
    }
}

impl <E> IndexMut<usize> for MetalVec<E> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let ptr = self.buf.contents() as *mut c_void;
        let ptr = ptr.cast::<E>();
        let mut data = unsafe {
            ptr.as_mut().unwrap()
        };

        data
    }
}

impl <E> IntoIterator for MetalVec<E> {
    type Item = E;

    type IntoIter = MetalVecIntoIter<E>;

    fn into_iter(self) -> Self::IntoIter {
        let t = Self::IntoIter {
            ptr: (self.buf.contents() as *const c_void).cast::<E>(),
            len: self.len,
            idx: 0,
        };

        t.into_iter()
    }
}

pub struct MetalVecIntoIter<E> {
    ptr: *const E,
    len: usize,
    idx: usize,
}

impl <E> Iterator for MetalVecIntoIter<E> {
    type Item = E;

    fn next(&mut self) -> Option<Self::Item> {
        let data = unsafe {
            let ptr = self.ptr.offset(self.idx as isize);
            ptr.read()
        };

        if self.idx < self.len {
            self.idx+=1;
            Some(data)
        } else {
            None
        }
    }
}


impl <E: Unit> Storage<E> for MetalGPU {
    type Vec = MetalVec<E>;

    fn try_alloc_len(&self, len: usize) -> Result<Self::Vec, Self::Err> {
        todo!()
    }

    fn try_alloc_ones(&self, len: usize) -> Result<Self::Vec, Self::Err> {
        todo!()
    }

    fn num_el(&self, st: Self::Vec) -> usize {
        todo!()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum MetalGpuError {
    SmallProblem
}

impl Display for MetalGpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl HasErr for MetalGPU {
    type Err = MetalGpuError;
    const Err: Self::Err = MetalGpuError::SmallProblem;
}

impl <E: Unit> ZerosTensor<E> for MetalGPU {
    fn try_zeros_from<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err> {
        let shape = *src.shape();

        let test_data = vec![0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15].iter().map(|ele| E::from_u32(*ele as u32)).collect::<Vec<E>>();

        let buffer = self.device.new_buffer_with_data(
            test_data.as_ptr().cast::<c_void>(), 
            (shape.num_elements() * size_of::<E>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // let buffer = self.device.new_buffer(shape.num_elements() as _, MTLResourceOptions::StorageModeShared);

        let buffer: MetalVec<E> = MetalVec {
            buf: buffer,
            len: shape.num_elements(),
            _marker: PhantomData
        };

        let data = Arc::new(RwLock::new(buffer));

        Ok(Tensor {
            id: unique_id(),
            device: self.clone(),
            tape: NoneTape,
            shape,
            data,
        })

    }
}