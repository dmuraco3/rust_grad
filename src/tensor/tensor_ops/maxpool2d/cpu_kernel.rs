use std::ops::{Index, IndexMut};

use crate::{dtypes::Unit, devices::cpu::CPU, shape::Dim};

use super::{MaxPool2DKernel, PADDING};

impl <E: Unit> MaxPool2DKernel<E> for CPU {
    fn forward<
        SrcY: Dim,
        SrcX: Dim,
        FilterY: Dim,
        FilterX: Dim,
        OutY: Dim,
        OutX: Dim,
        Stride: Dim
    >(
        &self,
        src: &crate::tensor::Tensor<(SrcY, SrcX), E, Self>,
        out: &mut crate::tensor::Tensor<(OutY, OutX), E, Self>,
        filter_shape: (FilterY, FilterX),
        stride: Stride,
        padding: PADDING,
    ) -> Result<(), Self::Err> {

        let mut out_inner = out.data.write().unwrap();

        
        let src_inner = src.data.read().unwrap();
        let out_shape = out.shape;

        let (src_inner, src_shape) = if (src.shape.0.size() % filter_shape.0.size(), src.shape.1.size() % filter_shape.1.size()) != (0, 0) {
            match padding {
                PADDING::VALID => {
                    let drop_y = src.shape.0.size() % filter_shape.0.size();
                    let drop_x = src.shape.1.size() % filter_shape.1.size();
                    // (Y, X)
                    let drop_src_shape = (src.shape.0.size()-drop_y, src.shape.1.size()-drop_x);

                    let mut drop_src: Vec<E> = Vec::new();
                    for el in src_inner.index(0..drop_src_shape.0 * src.shape.1.size()).chunks(src.shape.1.size()) {
                        drop_src.extend_from_slice(&el[0..drop_src_shape.1]);
                    }


                    assert_eq!(drop_src.len(), drop_src_shape.0*drop_src_shape.1);

                    (drop_src, drop_src_shape)
                },
                PADDING::SAME => {
                    let add_y = src.shape.0.size() % filter_shape.0.size();
                    let add_x = src.shape.1.size() % filter_shape.0.size();
                    
                    let add_src_shape = (src.shape.0.size()+add_y, src.shape.1.size()+add_x);

                    let mut add_src: Vec<E> = Vec::new();
                    for el in src_inner.index(0..src.shape.0.size() * src.shape.1.size()).chunks(src.shape.1.size()) {
                        add_src.extend_from_slice(el);
                        add_src.extend(vec![E::ZERO;add_x]);
                    }
                    add_src.extend(vec![E::ZERO;add_y*(src.shape.1.size()+add_x)]);

                    assert_eq!(add_src.len(), add_src_shape.0*add_src_shape.1);

                    (add_src, add_src_shape)
                },
            }
        } else {
            (src.data.read().unwrap().clone(), (src.shape.0.size(), src.shape.1.size()))
        };


        for out_y in 0..out_shape.0.size() {
            for out_x in 0..out_shape.1.size() {

                let mut max = E::ZERO;

                for filter_y in 0..filter_shape.0.size() {
                    for filter_x in 0..filter_shape.1.size() {
                        let l = src_inner.index((src_shape.1.size() * (out_y * stride.size() + filter_y)) + (out_x * stride.size() + filter_x) );

                        if *l > max {max = *l}

                    }
                }

                *out_inner.index_mut(out_shape.1.size() * out_y + out_x) = max;
            }
        }

        Ok(())
    }
}