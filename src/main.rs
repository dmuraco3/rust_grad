use std::{fs, io::{self, Read}, sync::{Arc, RwLock}, process::exit};

use rust_grad::{devices::cpu::CPU, tensor::{ZerosTensor, Tensor, tensor_ops::{matmul::{TryMatMul, MatMatKernel}, relu::TryReLU, pow::TryPow, sum::{TrySum, SumKernel}}, Watch, tape::{unique_id, NoneTape, SplitTape, PutTape}, RandTensor}, shape::{Rank2, Dim, Shape, Const, Rank1}};

// use rust_grad::{
//     tensor::{
//         Tensor,
//         tensor_ops::{
//             matmul::TryMatMul,
//             conv2d::TryConv2D,
//             maxpool2d::{
//                 TryMaxPool2D,
//                 PADDING
//             },
//             relu::TryReLU,
//             reshape::ReshapeTrait,
//             softmax::TrySoftmax,
//             sparse_categorical_crossentropy::TrySparseCategoricalCrossentropy
//         },
//         ZerosTensor,
//         RandTensor, OwnedTape, Gradients, unique_id
//     },
//     shape::Const,
//     devices::cpu::CPU
// };

const NUM_IMAGES: usize = 60_000;

fn read_training_data() -> (Vec<[f32;784]>, Vec<f32>) {
    let mut images: Vec<[f32;784]> = Vec::new();
    {
        let images_data_bytes = fs::File::open("/Users/dmuraco/rust_projects/rust_grad/data/mnist_dataset/train-images.idx3-ubyte").unwrap();
        let images_data_bytes = io::BufReader::new(images_data_bytes);
    
        let mut buffer_128 = [0_u8;16];
        images_data_bytes.get_ref().take(16).read_exact(&mut buffer_128).unwrap();
    
        {
            let mut img_buf = [0_u8;784];
        
            let mut img_buf_scaled = [0_f32;784];
        
            for _ in 0..NUM_IMAGES {
                images_data_bytes.get_ref().take(u64::try_from(28*28).unwrap()).read_exact(&mut img_buf).unwrap();
                img_buf.iter_mut().zip(img_buf_scaled.iter_mut()).for_each(|(x, y)| *y = (*x as f32) / 255.0 );
                images.push(img_buf_scaled);
            }
        }
    }

    let mut labels: Vec<f32> = Vec::new();
    {
        let labels_data_bytes = fs::File::open("/Users/dmuraco/rust_projects/rust_grad/data/mnist_dataset/train-labels.idx1-ubyte").unwrap();
        let labels_data_bytes = io::BufReader::new(labels_data_bytes);
        let mut buffer_64 = [0_u8;8];
        labels_data_bytes.get_ref().take(8).read_exact(&mut buffer_64).unwrap();
        {
            let mut label_buf = [0_u8];
            let mut label_buf_f32 = [0_f32];
            for _ in 0..NUM_IMAGES {
                labels_data_bytes.get_ref().take(u64::try_from(1).unwrap()).read_exact(&mut label_buf).unwrap();
                label_buf_f32[0] = label_buf[0] as f32;
                labels.push(label_buf_f32[0]);
            }
        }
    }

    (images, labels)
}


fn test_network() {
    let (images, labels) = read_training_data();
    const STEPS: usize = 100;
    const BATCHSIZE: usize = 128;
    let device = CPU::default();

    let mut w1: Tensor<Rank2<512, 784>, f32, _> = device.fill_rand_range(-1_f32..1_f32);
    let mut w2: Tensor<Rank2<128, 512>, f32, _> = device.fill_rand_range(-1_f32..1_f32);
    let mut w3: Tensor<Rank2<32, 128>, f32, _> = device.fill_rand_range(-1_f32..1_f32);
    let mut l1: Tensor<Rank2<10, 32>, f32, _> = device.fill_rand_range(-1_f32..1_f32);
    
    for i in 0..STEPS {
        let distrubution: Tensor<(Const<BATCHSIZE>,), i32, _> = device.fill_rand_range(0..images.len() as i32);
        let mut x: Vec<[f32;784]> = Vec::new();
        let mut y: Vec<f32> = Vec::new();
        // let mut l1: Tensor<(Const<)>
        for i in distrubution.data.read().unwrap().iter() {
            x.push(images[*i as usize])
        }
        for i in distrubution.data.read().unwrap().iter() {
            y.push(labels[*i as usize])
        }
        // Forward pass of the network
        for (i, j) in x.iter().zip(y.iter()) {
            let actual: Tensor<(Const<10>, ), f32, _> = device.zeros();
            actual.data.write().unwrap()[*j as usize] = *j;
            
            let mut x: Tensor<Rank1<784>, f32, CPU> = device.zeros();
            x.copy_from_slice(i);

            let x: Tensor<Rank1<512>, _, _, _> = w1.watch_leaky().matmul(x);


            return
        }
    }
}

fn main() {
    test_network();
}