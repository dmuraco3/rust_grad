use std::{fs, io::{self, Read}, sync::{Arc, RwLock}, time::Instant};
use colored::Colorize;

use rand::Rng;
use rust_grad::{devices::{cpu::CPU, metal::MetalGPU}, tensor::{ZerosTensor, Tensor, tensor_ops::{matmul::TryMatMul, relu::TryReLU, cross_entropy::TryCrossEntropy, utilities::backward::BackwardPropagate, softmax::TrySoftmax, add::TryAdd}, Watch, tape::{SplitTape, PutTape}, RandTensor}, shape::{Rank2, Const, Rank1}, nn::optim::ADAM};

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

fn test_medium_network() {
    let (images, labels) = read_training_data();
    
    
    const STEPS: usize = 400;
    const BATCHSIZE: usize = 128;

    const L1: f32 = 784_f32;
    const L2: f32 = 128_f32;
    const L3: f32 = 64_f32 ;
    const L4: f32 = 10_f32 ;

    

    let device: CPU = CPU::default();

    let b1: f32 = (6_f32 / (L1+L2)).sqrt();
    let b2: f32 = (6_f32 / (L2+L3)).sqrt();
    let b3: f32 = (6_f32 / (L3+L4)).sqrt();

    let w1: Tensor<Rank2<{L2 as usize}, {L1 as usize}>, f32, _> = device.fill_rand_range(-b1..b1);
    let w2: Tensor<Rank2<{L3 as usize}, {L2 as usize}>, f32, _> = device.fill_rand_range(-b2..b2);
    let w3: Tensor<Rank2<{L4 as usize}, {L3 as usize}>, f32, _>  = device.fill_rand_range(-b3..b3);
    
    let bias_1: Tensor<Rank1<{L2 as usize}>, f32, _> = device.zeros();
    let bias_2: Tensor<Rank1<{L3 as usize}>, f32, _> = device.zeros();
    let bias_3: Tensor<Rank1<{L4 as usize}>, f32, _> = device.zeros();

    let mut optimizer: ADAM<f32, _> = ADAM::new(&device, &[
        &w1, 
        &w2, 
        &w3, 
        &bias_1, 
        &bias_2,
        &bias_3,
    ]);


    let mut losses = Vec::new();

    let mut converged = false;

    let start = Instant::now();

    for i_step in 0..STEPS {
        let mut rng = rand::thread_rng();
        // let distribution: Tensor<(Const<BATCHSIZE>,), i32, _> = device.fill_rand_range(0..images.len() as i32);
        let mut x: Vec<[f32;784]> = Vec::new();
        let mut y: Vec<f32> = Vec::new();
        
        for _batch in 0..BATCHSIZE {
            let rand_idx = rng.gen_range(0..images.len());
            x.push(images[rand_idx]);
            y.push(labels[rand_idx]);
        }
        
        // Forward pass of the network
        for (batch, (image_flat, label_index)) in x.iter().zip(y.iter()).enumerate() {
            let labels: Tensor<Rank1<10>, f32, _> = device.zeros();
            labels.data.write().unwrap()[*label_index as usize] = 1_f32;
            let mut image: Tensor<Rank1<784>, f32, CPU> = device.zeros();
            image.copy_from_slice(image_flat); 

            let x = image.watch_leaky().matmul(w1.clone()).add(bias_1.clone()).relu();
            let x = x.matmul(w2.clone()).add(bias_2.clone()).relu();
            let x = x.matmul(w3.clone()).add(bias_3.clone());
            let loss = x.try_cross_entropy(labels).unwrap();
    
            if loss.data.read().unwrap().iter().sum::<f32>().is_nan() {converged = true;break;} 

            losses.push(loss.data.read().unwrap()[0]);
            let grads = loss.backward();
            optimizer.step(grads);
        }

        if converged {
            break;
        }

        println!("{:?}", losses.iter().fold(0_f32, |acc, x| {
            acc + x
        }) / losses.len() as f32);
    }
    println!("time to train: {:?}", start.elapsed());

    // evaluate output of network
    let actual: Tensor<(Const<10>, ), f32, _> = device.zeros();

    actual.data.write().unwrap()[labels[2127] as usize] = 1_f32;
    
    let mut x: Tensor<Rank1<784>, f32, CPU> = device.zeros();
    x.copy_from_slice(&images[2127]);

    let mut cool_picture: Tensor<Rank2<28, 28>, f32, CPU> = device.zeros();
    cool_picture.data = Arc::new(RwLock::new(x.data.read().unwrap().to_owned()));

    for x in cool_picture.data.read().unwrap().chunks(28) {
        for y in x {
            let shade = (y * 255.0) as u8;
            print!("{}", "▊▊".custom_color(colored::CustomColor::new(shade, shade, shade)))
        }
        println!();
    }
    let x = x.watch_leaky().matmul(w1.clone()).add(bias_1.clone()).relu();
    let x = x.matmul(w2.clone()).add(bias_2.clone()).relu();
    let (x, x_tape) = x.matmul(w3.clone()).add(bias_3.clone()).split_tape();
    let loss = x.clone().put_tape(x_tape).try_cross_entropy(actual.clone()).unwrap();
    let soft = x.softmax();

    println!("{:?}", soft.data.read().unwrap());
    println!("{:?}", actual.data.read().unwrap());
    println!("loss: {}", loss.data.read().unwrap().first().unwrap())
}

fn test_tensor() {
    let dev = CPU::default();
    
    // let src: Tensor<Rank1<1000>, f32, _> = dev.fill_rand_range(0_f32..0.9_f32);
    let src: Tensor<(Const<8>,), f32, _> = dev.from_array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]);

    let weights: Tensor<Rank2<8, 8>, f32, _> = dev.fill_rand_range(-0.5_f32..0.5_f32);

    let biases: Tensor<(Const<8>,), f32, CPU> = dev.zeros();

    let mut labels = dev.zeros();
    labels.data.write().unwrap()[0]=1_f32;


    let mut optim = ADAM::new(&dev, &[&weights, &biases]);

    for _ in 0..1 {
        let x = weights.watch_leaky().matmul(src.clone()).add(biases.clone()).relu();

        println!("{:?}", x.data.read().unwrap());
        
        let x_id = x.id.clone();

        let (cross_entropy, cross_entropy_tape) = x.try_cross_entropy(labels.clone()).unwrap().split_tape();
    
        // println!("{}: {}", iii, cross_entropy.data.read().unwrap()[0]);
        // if cross_entropy.data.read().unwrap()[0] == 0.0 {
        //     exit(-1)
        // }
        // cross_entropy.tape.operations.sort_by(|a, b| b.0.cmp(&a.0));
        // for op in cross_entropy.tape.operations.into_iter() {
        //     op.1(&mut cross_entropy.tape.gradients).unwrap();
        // }
        let grads = cross_entropy.clone().put_tape(cross_entropy_tape).backward();

        println!("{:?}", grads.get_grad_ref(&biases.id));
        println!();
        println!("bias before optim: {:?}", biases.data.read().unwrap());
        optim.step(grads);

        println!("{:?}", biases.data.read().unwrap());
    
        // println!("{}", cross_entropy.tape.gradients.get(&weights).unwrap());
    }

    // println!("actual_grad : {:?}", actual_grad.data.read().unwrap());
    // println!("eq: {}", src_grad.allclose(&actual_grad, None, None));
}

fn main() {
    // test_big_network();    
    // test_medium_network();

    let device = MetalGPU::default();

    let ten: Tensor<Rank2<4, 4>, f32, MetalGPU> = device.zeros();

    for x in ten.data.read().unwrap().clone().into_iter() {
        println!("{}", x);
    }


}