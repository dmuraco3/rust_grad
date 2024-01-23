use std::{time::Instant, mem::size_of, borrow::BorrowMut};

use metal::{objc::rc::autoreleasepool, mps::{matrix::{Matrix, MatrixDescriptor, MatrixDescriptorRef, MPSMatrixMultiplication, MatrixMultiplication, encode_gemm_mbuffers, MatrixBuffer}, MPSDataType, Float32}, Buffer};

use crate::{dtypes::Unit, shape::Dim, devices::metal::MetalGPU, tensor::{Tensor, tape::Gradients, ZerosTensor, tensor_ops::matmul::MatVecKernel}};

use super::super::MatMatKernel;

const LIB_DATA: &[u8] = include_bytes!("matmul.metallib");

struct MetalState {
    pub queue: metal::CommandQueue,
    pub pipeline: metal::ComputePipelineState,
}

impl <E: Unit> MatVecKernel<E> for MetalGPU {
    fn forward<I: Dim, J: Dim>(
        &self,
        lhs: &Tensor<(I, J), E, Self>,
        rhs: &Tensor<(J, ), E, Self>,
    ) -> Result<Tensor<(I,), E, Self>, Self::Err> {
        let (i, j) = lhs.shape;

        let result = self.try_zeros_from(&(i,)).unwrap();

        let lhs_buffer = &lhs.data.read().unwrap().buf;
        let rhs_buffer = &rhs.data.read().unwrap().buf;

        let res_data = result.data.write().unwrap();
        let res_buffer = &res_data.buf;

        let lhs_mps_matrix: MatrixBuffer<Float32> = MatrixBuffer::from_buffer(lhs_buffer.to_owned(), i.size() as u64, j.size() as u64);
        let rhs_mps_matrix: MatrixBuffer<Float32> = MatrixBuffer::from_buffer(rhs_buffer.to_owned(), j.size() as u64, 1);
        let mut res_mps_matrix: MatrixBuffer<Float32> = MatrixBuffer::from_buffer(res_buffer.to_owned(), i.size() as u64, 1);

        autoreleasepool(|| {
            let queue = self.device.new_command_queue();
    
            let command_buffer = queue.new_command_buffer();

            encode_gemm_mbuffers(&self.device, command_buffer, false, false, &lhs_mps_matrix, &rhs_mps_matrix, &mut res_mps_matrix, 1.0, 0.0, None);

            let start = Instant::now();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let elapsed = start.elapsed();
            println!("gpu time {:?}", elapsed);
        });

        drop(res_data);

        Ok(result)
    }

    fn backward<I: Dim, J: Dim>(
        &self,
        lhs: &Tensor<(I, J), E, Self>,
        rhs: &Tensor<(J, ), E, Self>,
        grads: &mut Gradients<E, Self>,
        out: &Tensor<(I,), E, Self>,
    ) -> Result<(), Self::Err> {
        todo!()
    }
}

impl <E: Unit> MatMatKernel<E> for MetalGPU {
    fn forward<I: Dim, J: Dim, K: Dim>(
        &self, 
        lhs: &Tensor<(I,J), E, Self>,
        rhs: &Tensor<(J,K), E, Self>,
    ) -> Result<Tensor<(I,K), E, Self>, Self::Err> {
        let (i, j) = lhs.shape;
        let k = rhs.shape.1;

        let result: Tensor<(I, K), E, MetalGPU> = self.try_zeros_from(&(i, k)).unwrap();

        let lhs_buffer = &lhs.data.read().unwrap().buf;
        let rhs_buffer = &rhs.data.read().unwrap().buf;
        let binding = result.data.write().unwrap();
        let res_buffer= &binding.buf;

        let lhs_mps_matrix: MatrixBuffer<Float32> = MatrixBuffer::from_buffer(lhs_buffer.to_owned(), i.size() as u64, j.size() as u64);
        let rhs_mps_matrix: MatrixBuffer<Float32> = MatrixBuffer::from_buffer(rhs_buffer.to_owned(), j.size() as u64, k.size() as u64);
        let mut res_mps_matrix: MatrixBuffer<Float32> = MatrixBuffer::from_buffer(res_buffer.to_owned(), i.size() as u64, k.size() as u64);


        autoreleasepool(|| {
            let queue = self.device.new_command_queue();
    
            let command_buffer = queue.new_command_buffer();

            encode_gemm_mbuffers(&self.device, command_buffer, false, false, &lhs_mps_matrix, &rhs_mps_matrix, &mut res_mps_matrix, 1.0, 0.0, None);

            let start = Instant::now();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let elapsed = start.elapsed();
            println!("gpu time {:?}", elapsed);
        });

        drop(binding);
        
        return Ok(result)
    }

    fn backward<I: Dim, J: Dim, K:Dim>(
        &self,
        lhs: &Tensor<(I,J), E, Self>,
        rhs: &Tensor<(J,K), E, Self>,
        grads: &mut Gradients<E, Self>,
        out: &Tensor<(I, K), E, Self>
    ) -> Result<(), Self::Err> {
        todo!()
    }
}