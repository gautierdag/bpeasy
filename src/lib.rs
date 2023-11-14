use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyIterator, PyString};
extern crate regex;
use regex::Regex;
use std::collections::HashMap;

fn tokenize(text: &str, regex: &str) -> Vec<String> {
    // regex splits
    let re = Regex::new(regex).unwrap();
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

fn get_most_frequent_pair(
    tokenized_bytes: &mut Vec<Vec<Vec<u8>>>,
    max_token_length: usize,
) -> Option<(Vec<u8>, Vec<u8>)> {
    /*
    Calculate frequencies for each pair of bytes in all sentences and words
    Return the most frequent pair of bytes
    */

    let mut pair_freqs: HashMap<(Vec<u8>, Vec<u8>), u128> = HashMap::new();

    // Calculate frequencies for each pair of bytes in all sentences and words
    for sentence in tokenized_bytes {
        for word in sentence.windows(2) {
            if word[0].len() + word[1].len() > max_token_length {
                continue;
            }
            if let [a, b] = word {
                *pair_freqs.entry((a.to_vec(), b.to_vec())).or_insert(0) += 1;
            }
        }
    }
    // println!("{:?}", pair_freqs);
    let most_frequent_pair = pair_freqs.iter().max_by_key(|&(_, count)| count);
    println!("Most frequent pair: {:?}", most_frequent_pair);
    if most_frequent_pair.is_none() {
        return None;
    }
    let ((ref left, ref right), _count) = most_frequent_pair.unwrap();
    Some((left.clone(), right.clone()))
}

fn merge_frequent_pair(tokenized_bytes: &mut Vec<Vec<Vec<u8>>>, left: Vec<u8>, right: Vec<u8>) {
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
}

fn build_bpe_vocab(
    mut tokenized_bytes: Vec<Vec<Vec<u8>>>,
    max_token_length: usize,
    vocab_size: usize,
) -> HashMap<Vec<u8>, u64> {
    let mut vocab: HashMap<Vec<u8>, u64> = initialize_vocab_bytes();

    println!("{:?}", vocab);

    let mut num_token_added = 0;
    while num_token_added < vocab_size {
        println!("Iteration: {}", num_token_added);

        let most_frequent_pair = get_most_frequent_pair(&mut tokenized_bytes, max_token_length);
        if most_frequent_pair.is_none() {
            break;
        }
        let (left, right) = most_frequent_pair.unwrap();

        // Merge the most frequent pair in all sentences and words
        merge_frequent_pair(&mut tokenized_bytes, left.clone(), right.clone());

        let mut token = left.clone(); // Clone the first token
        token.extend(right); // Extend with the second token
                             // Now, combined_token contains the merged pair
        println!("Combined token: {:?}", token);

        // combine pair into a single token
        let token_str = String::from_utf8_lossy(&token);
        println!("Token added: {:?}", token_str);
        vocab.insert(token, vocab.len() as u64);

        num_token_added += 1;
    }
    // print_vocab_bytes(&vocab);
    vocab
}

// Train BPE from Iterator
#[pyfunction]
fn train_bpe(
    py: Python,
    iterator: PyObject,
    python_regex: &PyString,
    max_token_length: usize,
    vocab_size: usize,
) -> PyResult<PyObject> {
    let iterator = PyIterator::from_object(py, &iterator)?;
    let regex = python_regex.to_str()?;

    let mut tokenized_bytes: Vec<Vec<Vec<u8>>> = Vec::new();

    // split all text into tokens
    for item in iterator {
        let item: &PyString = item?.extract()?;
        let text = item.to_str()?;
        let tokens = tokenize(text, regex);
        let tokens_bytes = convert_to_tokenized_bytes(tokens);
        tokenized_bytes.extend(tokens_bytes);
    }

    let bpe_vocab = build_bpe_vocab(tokenized_bytes, max_token_length, vocab_size);
    let python_dict_out = PyDict::new(py);

    // convert bpe_vocab to python dict
    for (key, value) in bpe_vocab {
        let py_key = PyBytes::new(py, &key);
        python_dict_out.set_item(py_key, value)?;
    }

    Ok(python_dict_out.into())
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn bpeasy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_bpe, m)?)?;
    Ok(())
}

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let text = "Your text data here";
        let regex = r"([^\s]+)|(\s+)";
        let tokens = tokenize(text, regex);
        assert_eq!(tokens, vec!["Your", " ", "text", " ", "data", " ", "here"]);
    }

    #[test]
    fn test_all() {
        let text: &str = "\tYou hear £ £ £ here";
        let regex = r"([^\s]+)|(\s+)";
        let tokens = tokenize(text, regex);
        println!("{:?}", tokens);
        let tokenized_bytes = convert_to_tokenized_bytes(tokens);
        println!("{:?}", tokenized_bytes);

        let vocab_size = 10;
        let max_token_length = 128;
        let bpe_vocab = build_bpe_vocab(tokenized_bytes, max_token_length, vocab_size);
        println!("{:?}", bpe_vocab);
        // Output or use the encoded text
    }
}
