extern crate regex;
use regex::Regex;
use std::collections::HashMap;

fn tokenize(text: &str) -> Vec<String> {
    // regex splits
    let re = Regex::new(r"([^\s]+)|(\s+)").unwrap();
    re.find_iter(text)
        .map(|mat| mat.as_str().to_string())
        .collect()
}

fn convert_to_tokenized_bytes(tokenized_text: Vec<String>) -> Vec<Vec<Vec<u8>>> {
    let mut tokenized_bytes: Vec<Vec<Vec<u8>>> = Vec::new();
    for token in tokenized_text {
        let mut tokenized_byte: Vec<Vec<u8>> = Vec::new();
        for byte in token.bytes() {
            tokenized_byte.push(vec![byte]);
        }
        tokenized_bytes.push(tokenized_byte);
    }
    tokenized_bytes
}

fn initialize_vocab_bytes() -> HashMap<Vec<u8>, u64> {
    let mut vocab: HashMap<Vec<u8>, u64> = HashMap::new();
    for i in 0..255 {
        vocab.insert(vec![i], i as u64);
    }
    vocab
}

// fn print_vocab_bytes(vocab: &HashMap<Vec<u8>, u64>) {
//     // sort by value
//     let mut sorted_vocab: Vec<_> = vocab.iter().collect();
//     sorted_vocab.sort_by(|a, b| a.1.cmp(b.1));
//     for (key, value) in sorted_vocab {
//         // try to convert to string
//         let key_str = String::from_utf8_lossy(key);
//         println!("{:?}: {}", key_str, value);
//     }
// }
// fn merge_frequent_pair(
//     tokenized_bytes: &mut Vec<Vec<Vec<u8>>>,
//     most_frequent_pair: &(Vec<u8>, Vec<u8>),
// ) {
// }

fn build_bpe_vocab(
    mut tokenized_bytes: Vec<Vec<Vec<u8>>>,
    vocab_size: usize,
) -> HashMap<Vec<u8>, u64> {
    let mut vocab: HashMap<Vec<u8>, u64> = initialize_vocab_bytes();

    println!("{:?}", vocab);

    let mut num_token_added = 0;
    while num_token_added < vocab_size {
        println!("Iteration: {}", num_token_added);
        let mut pair_freqs: HashMap<(Vec<u8>, Vec<u8>), u64> = HashMap::new();

        // Calculate frequencies for each pair of bytes in all sentences and words
        for sentence in &tokenized_bytes {
            for word in sentence.windows(2) {
                if let [a, b] = word {
                    *pair_freqs.entry((a.to_vec(), b.to_vec())).or_insert(0) += 1;
                }
            }
        }
        // println!("{:?}", pair_freqs);
        let most_frequent_pair = pair_freqs.iter().max_by_key(|&(_, count)| count);
        println!("Most frequent pair: {:?}", most_frequent_pair);
        if most_frequent_pair.is_none() {
            break;
        } else {
            let ((ref left, ref right), _count)) = most_frequent_pair;
            let mut token = left.clone(); // Clone the first token
            token.extend(right); // Extend with the second token

            // Now, combined_token contains the merged pair
            println!("Combined token: {:?}", token);

            // Assuming you have identified the most frequent pair (most_frequent_pair)
            // Merge the most frequent pair in all sentences and words
            for sentence in tokenized_bytes.iter_mut() {
                let mut i = 0;
                while i < sentence.len() - 1 {
                    // Check if the current and next token form the most frequent pair
                    if sentence[i] == left.clone() && sentence[i + 1] == right.clone() {
                        // Merge the pair and replace the first element with the merged pair
                        let merged = [&sentence[i][..], &sentence[i + 1][..]].concat();
                        sentence[i] = merged;
                        // Remove the second element of the pair
                        sentence.remove(i + 1);
                        // Do not increment i, as we want to check the next pair starting from the current position
                    } else {
                        i += 1; // Move to the next token
                    }
                }
            }

            // combine pair into a single token
            let token_str = String::from_utf8_lossy(&token);
            println!("Token added: {:?}", token_str);
            vocab.insert(token, vocab.len() as u64);

            num_token_added += 1;
        }
    }
    // print_vocab_bytes(&vocab);
    vocab
}

// fn encode_with_bpe(text: &str, bpe_vocab: &HashMap<String, String>) -> Vec<String> {
//     // Implement encoding logic using BPE vocabulary
// }

fn main() {
    let text: &str = "\tYou hear £ here";

    let tokens = tokenize(text);
    println!("{:?}", tokens);
    let tokenized_bytes = convert_to_tokenized_bytes(tokens);
    println!("{:?}", tokenized_bytes);

    let vocab_size = 10;
    let bpe_vocab = build_bpe_vocab(tokenized_bytes, vocab_size);
    println!("{:?}", bpe_vocab);

    // let bpe_vocab = build_bpe_vocab(text, vocab_size);
    // let encoded_text = encode_with_bpe(text, &bpe_vocab);

    // Output or use the encoded text
}

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let text = "Your text data here";
        let tokens = tokenize(text);
        assert_eq!(
            tokens,
            vec![
                "Y", "o", "u", "r", " ", "t", "e", "x", "t", " ", "d", "a", "t", "a", " ", "h",
                "e", "r", "e"
            ]
        );
    }
}
