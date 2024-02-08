use std::{
    borrow::BorrowMut,
    ffi::c_void,
    fmt::Display,
    marker::PhantomData,
    mem::size_of,
    ops::{Index, IndexMut, Range},
    sync::{Arc, RwLock},
    time::Instant,
};

use metal::{
    objc::rc::autoreleasepool, Buffer, CommandQueue, ComputePipelineState, Device,
    MTLResourceOptions,
};
use rand::{
    distributions::{uniform::SampleUniform, Standard, Uniform},
    prelude::Distribution,
    rngs::StdRng,
    Rng, SeedableRng,
};

use crate::{
    dtypes::Unit,
    shape::{HasShape, Rank1, Rank2, Shape},
    storage::Storage,
    tensor::{
        tape::{unique_id, NoneTape},
        HasErr, RandTensor, Tensor, ZerosTensor,
    },
};

#[derive(Debug, Clone)]
pub struct MetalGPU {
    pub(crate) device: Device,
}

impl Default for MetalGPU {
    fn default() -> Self {
        let device = Device::system_default().expect("No metal device found");

        Self { device }
    }
}

impl MetalGPU {
    pub fn from_array<const S: usize, E: Unit>(&self, src: [E; S]) -> Tensor<Rank1<S>, E, Self> {
        let mut tensor: Tensor<Rank1<S>, E, MetalGPU> = self.zeros();

        let mut tensor_inner = tensor.borrow_mut().data.write().unwrap();

        for (idx, ele) in src.iter().enumerate() {
            *tensor_inner.index_mut(idx) = *ele;
        }

        drop(tensor_inner);

        tensor
    }

    pub fn from_2d_array<const Y: usize, const X: usize, E: Unit>(
        &self,
        src: [[E; X]; Y],
    ) -> Tensor<Rank2<X, Y>, E, Self> {
        let mut tensor: Tensor<Rank2<X, Y>, E, MetalGPU> = self.zeros();

        let mut tensor_inner = tensor.borrow_mut().data.write().unwrap();

        for (y, row) in src.iter().enumerate() {
            for (x, ele) in row.iter().enumerate() {
                *tensor_inner.index_mut(y * X + x) = *ele;
            }
        }

        drop(tensor_inner);

        tensor
    }

    pub fn from_vec_with_shape<S: Shape, E: Unit>(
        &self,
        src: Vec<E>,
        shape: S,
    ) -> Tensor<S, E, Self> {
        assert_eq!(
            shape.num_elements(),
            src.len(),
            "src vec does not have same number of elements as desired shape"
        );
        let mut tensor = self.try_zeros_from(&shape).unwrap();
        let mut tensor_inner = tensor.borrow_mut().data.write().unwrap();

        for (idx, ele) in src.iter().enumerate() {
            *tensor_inner.index_mut(idx) = *ele;
        }

        drop(tensor_inner);

        tensor
    }
}

/// Vector backed by GPU memory
#[derive(Clone, Debug)]
pub struct MetalVec<E> {
    pub buf: metal::Buffer,
    pub len: usize,

    _marker: PhantomData<E>,
}

impl<E> Index<usize> for MetalVec<E> {
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

impl<E> IndexMut<usize> for MetalVec<E> {
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

// impl <E> ZerosTensor<E> for MetalGPU {
//     fn try_zeros_from<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err> {
//         let num_elements = src.num_elements();
//         let zeros = vec![0;num_elements];
//         let buf = self.device.new_buffer_with_data(&zeros as *const c_void,num_elements * size_of::<E>(), MTLResourceOptions::StorageModeShared);
//     }
// }

impl<E> IntoIterator for MetalVec<E> {
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

impl<E> IntoIterator for &MetalVec<E> {
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

impl<E> Iterator for MetalVecIntoIter<E> {
    type Item = E;

    fn next(&mut self) -> Option<Self::Item> {
        let data = unsafe {
            let ptr = self.ptr.offset(self.idx as isize);
            ptr.read()
        };

        if self.idx < self.len {
            self.idx += 1;
            Some(data)
        } else {
            None
        }
    }
}

impl<E: Unit> Display for MetalVec<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[ ")?;
        for e in self.into_iter() {
            write!(f, "{} ", e)?;
        }
        writeln!(f, "]")?;
        Ok(())
    }
}

impl<E: Unit> Storage<E> for MetalGPU {
    type Vec = MetalVec<E>;

    fn try_alloc_len(&self, len: usize) -> Result<Self::Vec, Self::Err> {
        let buf = self.device.new_buffer(
            (len * size_of::<E>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let buf = MetalVec {
            buf,
            len,
            _marker: PhantomData,
        };

        Ok(buf)
    }

    fn try_alloc_ones(&self, len: usize) -> Result<Self::Vec, Self::Err> {
        let ones = vec![E::ONE; len];
        let buf = self.device.new_buffer_with_data(
            ones.as_ptr().cast::<c_void>(),
            (len * size_of::<E>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf = MetalVec {
            buf,
            len,
            _marker: PhantomData,
        };

        Ok(buf)
    }

    fn num_el(&self, st: Self::Vec) -> usize {
        let _t = st.len;
        todo!()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum MetalGpuError {
    SmallProblem,
}

impl Display for MetalGpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl HasErr for MetalGPU {
    type Err = MetalGpuError;
    const ERR: Self::Err = MetalGpuError::SmallProblem;
}

impl<E: Unit> ZerosTensor<E> for MetalGPU {
    fn try_zeros_from<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err> {
        let shape = *src.shape();
        // let buffer = self.device.new_buffer_with_data(
        //     test_data.as_ptr().cast::<c_void>(),
        //     (shape.num_elements() * size_of::<E>()) as u64,
        //     MTLResourceOptions::StorageModeShared,
        // );
        let buffer = self.device.new_buffer(
            (size_of::<E>() * shape.num_elements()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let buffer: MetalVec<E> = MetalVec {
            buf: buffer,
            len: shape.num_elements(),
            _marker: PhantomData,
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

impl<E: Unit + SampleUniform> RandTensor<E> for MetalGPU {
    fn try_fill_rand<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err>
    where
        Standard: Distribution<E>,
    {
        let shape = *src.shape();
        let out: Tensor<S::Shape, E, MetalGPU> = self.try_zeros_from(&shape).unwrap();

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

    fn try_fill_rand_range<S: HasShape>(
        &self,
        src: &S,
        range: Range<E>,
    ) -> Result<Tensor<S::Shape, E, Self>, Self::Err>
    where
        Standard: Distribution<E>,
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

pub struct MetalState {
    pub queue: CommandQueue,
    pub pipeline: ComputePipelineState,
}

impl MetalState {
    pub fn new_with_shader(device: &Device, library_data: &[u8], shader_name: &str) -> Self {
        let queue = device.new_command_queue();
        let lib = device.new_library_with_data(library_data).unwrap();
        let function = lib.get_function(shader_name, None).unwrap();

        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .unwrap();

        Self { queue, pipeline }
    }
}

impl MetalGPU {
    pub fn call_kernel(
        &self,
        library_data: &[u8],
        shader_name: &str,
        buffers: &[&Buffer],
        shape: (usize, usize, usize),
    ) -> Result<(), <MetalGPU as HasErr>::Err> {
        let device = &self.device;

        autoreleasepool(|| {
            let state = MetalState::new_with_shader(device, library_data, shader_name);

            let command_buffer = state.queue.new_command_buffer();
            let compute_encoder = command_buffer.new_compute_command_encoder();
            compute_encoder.set_compute_pipeline_state(&state.pipeline);

            for (i, buffer) in buffers.iter().enumerate() {
                compute_encoder.set_buffer(i as u64, Some(buffer), 0);
            }

            let w = state.pipeline.thread_execution_width();
            let h = if shape.1 == 1 {
                1
            } else {
                state.pipeline.max_total_threads_per_threadgroup() / w
            };

            let grid_size = metal::MTLSize::new(shape.0 as u64, shape.1 as u64, shape.2 as u64);
            let threadgroup_size = metal::MTLSize::new(w, h, 1);

            compute_encoder.dispatch_threads(grid_size, threadgroup_size);

            compute_encoder.end_encoding();
            command_buffer.commit();

            #[cfg(debug_assertions)]
            let start = Instant::now();

            command_buffer.wait_until_completed();

            #[cfg(debug_assertions)]
            {
                let elapsed = start.elapsed();
                println!(
                    "time to execute {} on Metal GPU: {:?}",
                    shader_name, elapsed
                );
            }
        });

        Ok(())
    }

    /// notice the plural in kernels
    /// this method initializes and calls multiple kernels
    pub fn call_kernels() -> Result<(), <MetalGPU as HasErr>::Err> {
        todo!();
    }
}
