use rust_grad::{tensor::Tensor, shape::{Rank2, Storage, Rank1}, dtypes::Unit};
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

pub struct Embedding <const VOCAB_SIZE: usize, const EMBED_SIZE: usize, E: Unit, D: Storage<E>> {
    pub w: Tensor<Rank2<VOCAB_SIZE, EMBED_SIZE>, E, D>,
}


pub struct Linear<const IN_FEATURES: usize, const OUT_FEATURES: usize, E: Unit, D: Storage<E>> {
    pub weight: Tensor<Rank2<OUT_FEATURES, IN_FEATURES>, E, D>
}

pub struct RMSNorm <const DIM: usize, E: Unit, D: Storage<E>> {
    pub w: Tensor<Rank1<DIM>, E, D>
}

pub struct Attention<const DIM: usize, const HEADS: usize, const KVHEADS: usize, const MAXCONTENT: usize, E: Unit, D: Storage<E>> {
    pub wq: Linear<DIM, DIM, E, D>,
    pub wk: Linear<DIM, DIM, E, D>,
    pub wv: Linear<DIM, DIM, E, D>,
    pub wo: Linear<DIM, DIM, E, D>,
}

pub struct FeedForward <const DIM: usize, const HIDDEN_DIM: usize, E: Unit, D: Storage<E>> {
    pub w1: Linear<DIM, HIDDEN_DIM, E, D>,
    pub w2: Linear<DIM, HIDDEN_DIM, E, D>,
    pub w3: Linear<DIM, HIDDEN_DIM, E, D>,
}

pub struct TransformerBlock<const DIM: usize, const HIDDEN_DIM: usize, const HEADS: usize, const KVHEADS: usize, const MAXCONTENT: usize, E: Unit, D: Storage<E>> {
    pub attention: Attention<DIM, HEADS, KVHEADS, MAXCONTENT, E, D>,
    pub feed_forward: FeedForward<DIM, HIDDEN_DIM, E, D>,
    pub attention_norm: RMSNorm<DIM, E, D>,
    pub ffn_norm: RMSNorm<DIM, E, D>
}

pub struct Transformer<const LAYERS: usize, const DIM: usize, const HIDDEN_DIM: usize, const HEADS: usize, const KVHEADS: usize, const MAXCONTENT: usize, const VOCAB_SIZE: usize, E: Unit, D: Storage<E>> {
    pub layers: [TransformerBlock<DIM, HIDDEN_DIM, HEADS, KVHEADS, MAXCONTENT, E, D>;LAYERS],
    pub norm: RMSNorm<DIM, E, D>,
    pub tok_embeddings: Embedding<VOCAB_SIZE, DIM, E, D>,
    pub out_w: Linear<DIM, VOCAB_SIZE, E, D>,
    pub freqs_cis: Tensor<
      >
}

fn main() {
    let tokenizer = Tokenizer::build(TOKENIZER_PATH);

    let tokens = tokenizer.encode_with_pieces("I saw a girl with a telescope.", true, false);
    let tokens = tokenizer.encode("I saw a girl with a telescope", true, false);
    for (x, token) in tokens.iter().enumerate() {
        println!("{} : {:?}", x, token);
    }
}