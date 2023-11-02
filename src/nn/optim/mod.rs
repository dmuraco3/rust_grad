use std::{default, sync::{Arc, RwLock, atomic::AtomicUsize}, time::{Instant, Duration}, ops::AddAssign};

use crate::{dtypes::FloatUnit, tensor::{tape::{Gradients, UniqueID}, Tensor, TensorLike}, shape::{Storage, TensorInnerLength}, devices::cpu::CPU};

pub struct ADAMConfig<E: FloatUnit> {
    step_size: E,
    beta1: E,
    beta2: E,
    epsilon: E,
}

impl <E:FloatUnit> Default for ADAMConfig<E> {
    fn default() -> Self {
        Self {
            step_size: E::from_f32(0.001),
            beta1: E::from_f32(0.9), 
            beta2: E::from_f32(0.999),
            epsilon: E::from_f32(1e-08),
        }
    }
}


pub struct ADAM<E: FloatUnit, D: Storage<E>> {
    pub moment1   : Vec<(UniqueID, Arc<RwLock<D::Vec>>)>             ,
    pub moment2   : Vec<(UniqueID, Arc<RwLock<D::Vec>>)>             ,
    pub trainable : Vec<(UniqueID, Arc<RwLock<D::Vec>>)> ,
    pub t         : u32                                  ,
    pub config    : ADAMConfig<E>                        ,
}


// impl <E: FloatUnit, D: Storage<E>> Default for ADAM<E, D> {
//     fn default() -> Self {
//         Self {
//             moment1: Vec::new(),
//             moment2: Vec::new(),
//             config: ADAMConfig::default(),
//             trainable: Vec::default(),
//         }
//     }
// }

impl <'a, E: FloatUnit, D: Storage<E>> ADAM<E, D> {
    pub fn new(device: &D, trainable: &[&dyn TensorLike<E, D>]) -> Self {
        let mut moment1 = Vec::with_capacity(trainable.len());
        let mut moment2 = Vec::with_capacity(trainable.len());
        let trainable = trainable.iter().map(|tensor| {
            let moment_alloc = device.try_alloc_len(tensor.get_num_el()).unwrap();
            moment1.push((tensor.get_id(), Arc::new(RwLock::new(moment_alloc.to_owned()))));
            moment2.push((tensor.get_id(), Arc::new(RwLock::new(moment_alloc.to_owned()))));

            (tensor.get_id(), tensor.get_data_ref())
        }).collect::<Vec<_>>();

        Self {
            moment1,
            moment2,
            trainable,
            t: 0,
            config: Default::default(),
        }
    }
}

impl <E: FloatUnit> ADAM<E, CPU> {
    pub fn step(&mut self,  grads: Gradients<E, CPU>) {
        // for i_i in 0..self.trainable.len() {
        //     let grad = grads.gradient_by_id.get(&self.trainable[i_i].0).unwrap();
        //     let mut trainable_vector = self.trainable[i_i].1.write().unwrap();
        //     let mut m1_vector = self.moment1[i_i].write().unwrap();
        //     let mut m2_vector = self.moment2[i_i].write().unwrap();
        //     for j_j in 0..trainable_vector.len() {
        //         unsafe {
        //             let g = grad[j_j];
        //             let t = trainable_vector.get_unchecked_mut(j_j);
        //             let m1 = m1_vector.get_unchecked_mut(j_j);
        //             let m2 = m2_vector.get_unchecked_mut(j_j);
        //             *m1 = b1*(*m1) + (E::ONE-b1)*(g); // calculating m1_t
        //             *m2 = b2*(*m2) + (E::ONE-b2)*(g.pow(2)); // calculating m2_t
                    
        //             // correcting bias
        //             let m1_hat: E = *m1 / (E::ONE - b1.pow(self.t as u16));
        //             let m2_hat = *m2 / (E::ONE - b2.pow(self.t as u16));
        //             *t = *t - step_size * (m1_hat / (m2_hat.sqrt() + epsilon));
        //         }
        //     }
        // }
        // iterate through everything
        
        self.t+=1;

        let b1 = self.config.beta1;
        let b2 = self.config.beta2;
        let step_size = self.config.step_size;
        let epsilon = self.config.epsilon;
        
        self.trainable.iter().zip(
            self.moment1.iter().zip(self.moment2.iter())
        ).for_each(|(
            (
                trainable_id,
                trainable_vector,
            ),(
                m1_vector,
                m2_vector,
            )
        )|{
            let grad = grads.gradient_by_id.get(trainable_id).unwrap();
            let mut trainable_vector = trainable_vector.write().unwrap();

            let mut m1_vector_write = m1_vector.1.write().unwrap();
            let mut m2_vector_write = m2_vector.1.write().unwrap();

            grad.iter().zip(trainable_vector.iter_mut())
            .zip(
                m1_vector_write.iter_mut().zip(m2_vector_write.iter_mut())
            ).for_each(|((
                grad_ele,
                trainable_ele
            ),(
                m1_ele,
                m2_ele
            ))| {
                // actually doing element-wise ADAM operations here

                if *grad_ele < E::from_f32(-10.0) || *grad_ele > E::from_f32(10.0) {
                    println!("found a wild ass gradient: {}", grad_ele);
                }
                // clip gradient
                let grad_ele = *grad_ele + E::EPSILON;
                // let grad_ele = grad_ele.max(E::from_f32(-10_f32));
                // let grad_ele = grad_ele.min(E::from_f32(10_f32));


                // calculating moments for current timestep
                *m1_ele = b1*(*m1_ele) + (E::ONE-b1)*(grad_ele); // calculating m1_t
                *m2_ele = b2*(*m2_ele) + (E::ONE-b2)*(grad_ele.pow(2)); // calculating m2_t

                
                // correcting bias
                let m1_hat: E = *m1_ele / (E::ONE - b1.pow(self.t as u16));
                let m2_hat: E = *m2_ele / (E::ONE - b2.pow(self.t as u16));
                
                // println!("{}", step_size * (m1_hat / (m2_hat.sqrt() + epsilon)));
                // print!("{esc}c", esc = 27 as char);
                
                *trainable_ele = *trainable_ele - step_size * (m1_hat / (m2_hat.sqrt() + epsilon));


            })
        });

    }

}