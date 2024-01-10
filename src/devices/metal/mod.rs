use std::{fmt::{Display, Debug}, sync::{Arc, RwLock}, marker::PhantomData, clone, ops::{IndexMut, Index, Range, Deref}, ffi::c_void, mem::size_of};

use metal::{DeviceRef, Device, MTLResourceOptions, objc::rc::autoreleasepool, Buffer};
use rand::{distributions::{uniform::SampleUniform, Standard, Uniform}, prelude::Distribution, Rng, rngs::StdRng, SeedableRng};

use crate::{shape::{Shape, Storage, HasShape}, dtypes::Unit, tensor::{HasErr, ZerosTensor, Tensor, tape::{unique_id, NoneTape}, RandTensor}};

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
    pub buf: metal::Buffer,
    pub len: usize,

    _marker: PhantomData<E>
}

impl <E> Index<usize> for MetalVec<E> {
    type Output = E;

    fn index(&self, index: usize) -> &Self::Output {
        let ptr = self.buf.contents() as *const c_void;

        
        let data = unsafe {
            let ptr = ptr.add(index * size_of::<E>()).cast::<E>();
            ptr.as_ref().unwrap()
        };
        
        data
    }
}

impl <E> IndexMut<usize> for MetalVec<E> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    
        let ptr = self.buf.contents() as *mut c_void;
        let data = unsafe {
            let ptr = ptr.cast::<E>();
            let ptr = ptr.offset(index as isize);
            // println!("{} || {}", self.buf.contents() as isize, ptr as isize);
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

impl <E> IntoIterator for &MetalVec<E> {
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

impl <E: Unit> Display for MetalVec<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f,"[ ")?;
        for e in self.into_iter() {
            write!(f, "{} ", e)?;
        }
        writeln!(f, "]")?;
        Ok(())
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
        // let buffer = self.device.new_buffer_with_data(
        //     test_data.as_ptr().cast::<c_void>(), 
        //     (shape.num_elements() * size_of::<E>()) as u64,
        //     MTLResourceOptions::StorageModeShared,
        // );
        let buffer = self.device.new_buffer((size_of::<E>() * shape.num_elements()) as u64, MTLResourceOptions::StorageModeShared);
        
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

impl <E: Unit + SampleUniform> RandTensor<E> for MetalGPU {
    fn try_fill_rand<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err> 
    where
        Standard: Distribution<E>,
    {
        let shape = *src.shape();
        let mut out: Tensor<S::Shape, E, MetalGPU> = self.try_zeros_from(&shape).unwrap();

        let mut out_buf = out.data.write().unwrap();

        {
            let mut rng = rand::thread_rng();
            for idx in 0..shape.num_elements() {
                *out_buf.index_mut(idx) = rng.gen();
            }
        }

        drop(out_buf);
        
        Ok(out)
    }

    fn try_fill_rand_range<S: HasShape>(&self, src: &S, range: Range<E>) -> Result<Tensor<S::Shape, E, Self>, Self::Err>
    where Standard: Distribution<E>
    {
        let shape = *src.shape();
        let out: Tensor<S::Shape, E, MetalGPU> = self.try_zeros_from(&shape).unwrap();

        let mut out_buf = out.data.write().unwrap();

        let between = Uniform::from(range);
        // let mut rng = rand::thread_rng();
        let mut rng = StdRng::seed_from_u64(1227 as u64);

        for idx in 0..shape.num_elements() {
            *out_buf.index_mut(idx) = between.sample(&mut rng);
        }

        drop(out_buf);
        

        Ok(out)
    }
}