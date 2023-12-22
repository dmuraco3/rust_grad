use rust_grad::{tensor::{Tensor, ZerosTensor, Arange}, shape::{Rank2, Storage, Rank1, Rank3, Rank4, Const, Shape}, dtypes::Unit};
use sentencepiece::{SentencePieceProcessor, PieceWithId, SentencePieceError};
use sentencepiece_sys::spp_decode_piece_ids;


const TOKENIZER_PATH: &str = "/Users/dmuraco/llama/tokenizer.model";

struct Tokenizer {
    pub model: SentencePieceProcessor
}

impl Tokenizer {
    pub fn build(path: &str) -> Self {
        Self {
            model: SentencePieceProcessor::open(path).unwrap()
        }
    }

    /// Encodes a string into a vector of token IDs.
    /// 
    /// # Arguments
    /// 
    /// * `sentence` - The input string to be encoded.
    /// * `bos` - Whether to prepend the beginning-of-sequence token.
    /// * `eos` - Whether to append the end-of-sequence token.
    /// 
    pub fn encode(&self, sentence: &str, bos: bool, eos: bool) -> Vec<u32> {
        let tokens = self.model.encode(sentence).unwrap();
        let mut tokens = tokens.iter().map(|tok| tok.id).collect::<Vec<u32>>();

        if bos {
            let bos_id = self.model.bos_id().unwrap();
            tokens.insert(0, bos_id);
        }
        
        if eos {
            let eos_id = self.model.eos_id().unwrap();
            tokens.push(eos_id);
        }

        tokens
    }

    pub fn encode_with_pieces(&self, sentence: &str, bos: bool, eos: bool) -> Vec<String> {
        let tokens = self.model.encode(sentence).unwrap();
        let mut tokens = tokens.iter().map(|tok| tok.piece.clone()).collect::<Vec<String>>();

        if bos {
            let bos_id = self.model.bos_id().unwrap();
            let bos_decoded = self.model.decode_piece_ids(&[bos_id]).unwrap();
            // let bos_encoded_piece = self.model.encode(&bos_decoded).unwrap()[0].piece.clone();
            tokens.insert(0, bos_decoded);
        }
        if eos {
            let eos_id = self.model.eos_id().unwrap();
            let eos_decoded = self.model.decode_piece_ids(&[eos_id]).unwrap();
            println!("eos: {:?}", eos_decoded);
            // let eos_encoded_piece = self.model.encode(&eos_encoded).unwrap()[0].piece.clone();
            tokens.push(eos_decoded);
        }

        let eos_id = self.model.eos_id().unwrap();
        let eos_decoded = self.model.decode_piece_ids(&[eos_id]).unwrap();
        println!("eos: {:?}", eos_id);

        let fucker = self.model.to_serialized_proto();
        println!("fucker: {}", fucker[2]);

        tokens
    }

    /// Decodes a vector of token IDs into a string.
    /// 
    /// # Arguments
    /// 
    /// * `tokens` - The vector of token IDs to be decoded.
    pub fn decode(&self, tokens: Vec<u32>) -> String {
        self.model.decode_piece_ids(tokens.as_slice()).unwrap()
    }
}

#[derive(Clone)]
pub struct Embedding <const VOCAB_SIZE: usize, const EMBED_SIZE: usize, E: Unit, D: Storage<E>> {
    pub w: Tensor<Rank2<VOCAB_SIZE, EMBED_SIZE>, E, D>,
}

impl
    <const VOCAB_SIZE: usize, const EMBED_SIZE: usize, E, D>
Embedding
    <VOCAB_SIZE, EMBED_SIZE, E, D>
where 
    E: Unit,
    D: Storage<E> + ZerosTensor<E>
{
    pub fn init_with_device(device: &D) -> Self {
        Self {
            w: device.zeros()
        }
    }
}

#[derive(Clone)]
pub struct Linear<const IN_FEATURES: usize, const OUT_FEATURES: usize, E: Unit, D: Storage<E>> {
    pub weights: Tensor<Rank2<OUT_FEATURES, IN_FEATURES>, E, D>
}

impl 
    <const IN_FEATURES: usize, const OUT_FEATURES: usize, E, D>
Linear
    <IN_FEATURES, OUT_FEATURES, E, D>
where 
    E: Unit,
    D: Storage<E> + ZerosTensor<E>
{
    pub fn init_with_device(device: &D) -> Self {
        let weights: Tensor<Rank2<OUT_FEATURES, IN_FEATURES>, E, D> = device.zeros();
        Self {
            weights
        }
        
    }
}

#[derive(Clone)]
pub struct RMSNorm <const DIM: usize, E: Unit, D: Storage<E>> {
    pub w: Tensor<Rank1<DIM>, E, D>
}

impl 
    <const DIM: usize, E, D>
RMSNorm
    <DIM, E, D>
where
    E: Unit,
    D: Storage<E> + ZerosTensor<E>
{
    pub fn init_with_device(device: &D) -> Self {
        Self {
            w: device.zeros(),
        }
    }
}

#[derive(Clone)]
pub struct FeedForward <const DIM: usize, const HIDDEN_DIM: usize, E: Unit, D: Storage<E>> {
    pub w1: Linear<DIM, HIDDEN_DIM, E, D>,
    pub w2: Linear<DIM, HIDDEN_DIM, E, D>,
    pub w3: Linear<DIM, HIDDEN_DIM, E, D>,
}

impl
    <const DIM: usize, const HIDDEN_DIM: usize, E, D>
FeedForward
    <DIM, HIDDEN_DIM, E, D>
where
    E: Unit,
    D: Storage<E> + ZerosTensor<E>
{
    pub fn init_with_device(device: &D) -> Self {
        Self {
            w1: Linear::init_with_device(device),
            w2: Linear::init_with_device(device),
            w3: Linear::init_with_device(device),
        }
    }
}

#[derive(Clone)]
pub struct Attention <const DIM: usize, const HEADS: usize, const KVHEADS: usize, const MAXCONTENT: usize, E: Unit, D: Storage<E>> {
    pub wq: Linear<DIM, DIM, E, D>,
    pub wk: Linear<DIM, DIM, E, D>,
    pub wv: Linear<DIM, DIM, E, D>,
    pub wo: Linear<DIM, DIM, E, D>,
}

impl
    <const DIM: usize, const HEADS: usize, const KVHEADS: usize, const MAXCONTENT: usize, E, D>
Attention 
    <DIM, HEADS, KVHEADS, MAXCONTENT, E, D>
where
    E: Unit,
    D: Storage<E> + ZerosTensor<E>
{
    pub fn init_with_device(device: &D) -> Self {
        Self {
            wq: Linear::init_with_device(device),
            wk: Linear::init_with_device(device),
            wv: Linear::init_with_device(device),
            wo: Linear::init_with_device(device)
        }
    }
}

#[derive(Clone)]
pub struct TransformerBlock<const DIM: usize, const HIDDEN_DIM: usize, const HEADS: usize, const KVHEADS: usize, const MAXCONTENT: usize, E: Unit, D: Storage<E>> {
    pub attention: Attention<DIM, HEADS, KVHEADS, MAXCONTENT, E, D>,
    pub feed_forward: FeedForward<DIM, HIDDEN_DIM, E, D>,
    pub attention_norm: RMSNorm<DIM, E, D>,
    pub ffn_norm: RMSNorm<DIM, E, D>
}

impl
    <const DIM: usize, const HIDDEN_DIM: usize, const HEADS: usize, const KVHEADS: usize, const MAXCONTENT: usize, E, D>
TransformerBlock
    <DIM, HIDDEN_DIM, HEADS, KVHEADS, MAXCONTENT, E, D>
where
    E: Unit,
    D: Storage<E> + ZerosTensor<E>
{
    pub fn init_with_device(device: &D) -> Self {
        Self {
            attention: Attention::init_with_device(device),
            feed_forward: FeedForward::init_with_device(device),
            attention_norm: RMSNorm::init_with_device(device),
            ffn_norm: RMSNorm::init_with_device(device),
        }
    }
}

/// Improves efficiency of RoPE by pre-generating 
pub fn generate_rotary_matrix<E: Unit, D: Storage<E> + ZerosTensor<E>>(dim: usize, end: usize, theta: E, device: &D) -> Tensor<(Const<2>, usize), E, D> {
    let mut rotary_i = device.try_zeros_from(&(dim, )).unwrap();
    { // too tired to write a generic arange function right now
        let mut data = rotary_i.data.write().unwrap();
        for (i, ii) in (0..rotary_i.shape.num_elements()).zip((0..dim).step_by(2)) {
            data[i] = E::from_usize(ii);
        }
    }

    let mut rotary_matrix = device.try_zeros_from(&(Const::<2>, dim)).unwrap();

    rotary_matrix
} 

#[derive(Clone)]
pub struct Transformer<const LAYERS: usize, const DIM: usize, const HIDDEN_DIM: usize, const HEADS: usize, const KVHEADS: usize, const MAXCONTENT: usize, const VOCAB_SIZE: usize, E: Unit, D: Storage<E>> {
    pub layers: Vec<TransformerBlock<DIM, HIDDEN_DIM, HEADS, KVHEADS, MAXCONTENT, E, D>>,
    pub norm: RMSNorm<DIM, E, D>,
    pub tok_embeddings: Embedding<VOCAB_SIZE, DIM, E, D>,
    pub out_w: Linear<DIM, VOCAB_SIZE, E, D>,
    pub rotary_matrix: Tensor<(Const<2>, usize), E, D>,
}

impl 
    <const LAYERS: usize, const DIM: usize, const HIDDEN_DIM: usize, const HEADS: usize, const KVHEADS: usize, const MAXCONTENT: usize, const VOCAB_SIZE: usize, E, D>
Transformer
    <LAYERS, DIM, HIDDEN_DIM, HEADS, KVHEADS, MAXCONTENT, VOCAB_SIZE, E, D>
where
    E: Unit,
    D: Storage<E> + ZerosTensor<E>
{
    pub fn init_with_device(device: &D) -> Self {
        Self {
            layers: vec![TransformerBlock::init_with_device(device);LAYERS],
            norm: RMSNorm::init_with_device(device),
            tok_embeddings: Embedding::init_with_device(device),
            out_w: Linear::init_with_device(device),
            rotary_matrix: generate_rotary_matrix(DIM / HEADS, MAXCONTENT * 2, E::from_u32(10_000), device),
        }
    }

    pub fn forward(&mut self) {

    }
}

fn main() {
    let tokenizer = Tokenizer::build(TOKENIZER_PATH);

    let tokens = tokenizer.encode_with_pieces("I saw a girl with a telescope.", true, false);
    let tokens = tokenizer.encode("I saw a girl with a telescope", true, false);
    for (x, token) in tokens.iter().enumerate() {
        println!("{} : {:?}", x, token);
    }
}