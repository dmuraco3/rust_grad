use std::time::Instant;

use metal::{objc::rc::autoreleasepool, mps::{matrix::{encode_gemm_mbuffers, MatrixBuffer}, Float32}};

use crate::{dtypes::Unit, shape::{Dim, Shape}, devices::metal::MetalGPU, tensor::{Tensor, tape::Gradients, ZerosTensor, tensor_ops::matmul::MatVecKernel}};

use super::super::MatMatKernel;

const LIB_DATA: &[u8] = include_bytes!("matmul.metallib");

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

            encode_gemm_mbuffers(&self.device, command_buffer, false, false, &lhs_mps_matrix, &rhs_mps_matrix, &mut res_mps_matrix, 1.0, 0.0, None).unwrap();

            let start = Instant::now();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let elapsed = start.elapsed();
            #[cfg(debug_assertions)]
            println!("time to execute matvec gemm on Metal GPU: {:?}", elapsed);
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

        let (i, j) = lhs.shape.clone();

        grads.try_alloc_for((&lhs.device, lhs.id.clone(), lhs.shape.num_elements()))?;
        grads.try_alloc_for((&rhs.device, rhs.id.clone(), rhs.shape.num_elements()))?;
        grads.try_ones_for((&out.device, out.id.clone(), out.shape.num_elements()))?;

        let lhs_grad = grads.get_grad_mut(&lhs.id);
        let lhs_grad_buf = &lhs_grad.buf.to_owned();

        let rhs_grad = grads.get_grad_mut(&rhs.id);
        let rhs_grad_buf = &rhs_grad.buf.to_owned();

        let out_grad = grads.get_grad_ref(&out.id);
        let out_grad_buf = &out_grad.buf;

        let buffers = &[
            &lhs.data.read().unwrap().buf,
            &rhs.data.read().unwrap().buf,
            &out.data.read().unwrap().buf,
            &lhs_grad_buf,
            &rhs_grad_buf,
            &out_grad_buf,
        ];

        
        self.call_kernel(LIB_DATA, "matvec_backward", buffers, (i.size(), j.size(), 1))?;

        Ok(())
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

            encode_gemm_mbuffers(&self.device, command_buffer, false, false, &lhs_mps_matrix, &rhs_mps_matrix, &mut res_mps_matrix, 1.0, 0.0, None).unwrap();

            let start = Instant::now();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let elapsed = start.elapsed();
            println!("gpu time {:?}", elapsed);
        });

        drop(binding);
        
        return Ok(result)
    }

    #[allow(unused_variables)]
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