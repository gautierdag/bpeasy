use fancy_regex::Regex;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyIterator, PyString};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;

fn tokenize(text: &str, pattern: &str) -> Vec<Vec<Vec<u8>>> {
    let regex = Regex::new(pattern);

    let mut tokenized_bytes: Vec<Vec<Vec<u8>>> = Vec::new();

    for match_result in regex.expect(pattern).find_iter(text) {
        match match_result {
            Ok(token) => {
                let mut tokenized_byte: Vec<Vec<u8>> = Vec::new();
                for byte in token.as_str().bytes() {
                    tokenized_byte.push(vec![byte]);
                }
                tokenized_bytes.push(tokenized_byte);
            }
            Err(e) => {
                println!("Error: {:?}", e);
                break;
            }
        }
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

    // Calculate frequencies for each pair of bytes in all sentences and words
    // uses mutex to allow parallel processing through rayon
    let mut pair_freqs: HashMap<(Vec<u8>, Vec<u8>), u128> = HashMap::new();

    // Calculate frequencies for each pair of bytes in all sentences and words
    // NOTE: Could be parallelized over sentences
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

    // let pair_freqs: Mutex<HashMap<(Vec<u8>, Vec<u8>), u128>> = Mutex::new(HashMap::new());
    // tokenized_bytes.par_iter().for_each(|sentence| {
    //     let mut local_freqs = HashMap::new();
    //     for word in sentence.windows(2) {
    //         if word[0].len() + word[1].len() > max_token_length {
    //             continue;
    //         }
    //         if let [a, b] = word {
    //             *local_freqs.entry((a.to_vec(), b.to_vec())).or_insert(0) += 1;
    //         }
    //     }

    //     let mut global_freqs = pair_freqs.lock().unwrap();
    //     for (pair, count) in local_freqs {
    //         *global_freqs.entry(pair).or_insert(0) += count;
    //     }
    // });
    // let pair_freqs = pair_freqs.into_inner().unwrap();
    let most_frequent_pair = pair_freqs.iter().max_by_key(|&(_, count)| count);
    if most_frequent_pair.is_none() {
        return None;
    }
    let ((ref left, ref right), _count) = most_frequent_pair.unwrap();

    println!(
        "Most frequent pair: {:?} and count {}",
        (left, right),
        _count
    );
    Some((left.clone(), right.clone()))
}

fn merge_frequent_pair(tokenized_bytes: &mut Vec<Vec<Vec<u8>>>, left: Vec<u8>, right: Vec<u8>) {
    // Merge the most frequent pair in all sentences and words
    // NOTE: Could be parallelized over sentences
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
    iterator: &PyIterator,
    python_regex: &PyString,
    max_token_length: usize,
    vocab_size: usize,
) -> PyResult<PyObject> {
    let regex = python_regex.to_str()?;

    println!("STARTING BPEasy training");
    let num_threads = rayon::current_num_threads();
    println!("Number of threads: {}", num_threads);

    // validate inputs
    if max_token_length < 2 {
        return Err(exceptions::PyValueError::new_err(
            "max_token_length must be greater than 1",
        ));
    }
    if vocab_size < 1 {
        return Err(exceptions::PyValueError::new_err(
            "vocab_size must be greater than 0",
        ));
    }
    if regex.is_empty() {
        return Err(exceptions::PyValueError::new_err("regex cannot be empty"));
    }

    // let mut tokenized_bytes: Vec<Vec<Vec<u8>>> = Vec::new();
    // let tokenized_bytes = Mutex::new(Vec::new());

    // Extract strings from Python iterator and store them in a Rust Vec for parallel processing
    // let strings: Vec<&str> = iterator
    //     .filter_map(|item_result| {
    //         item_result.ok().and_then(|item| {
    //             item.extract::<&PyString>()
    //                 .ok()
    //                 .and_then(|py_string| py_string.to_str().ok())
    //         })
    //     })
    //     .collect();

    // // split all text into tokens
    // strings.par_iter().for_each(|text| {
    //     if !text.is_empty() {
    //         // println!("Text: {:?}", text);
    //         let tokens_bytes = tokenize(text, regex);
    //         // Lock the mutex and extend the vector
    //         let mut tokenized_bytes_lock = tokenized_bytes.lock().unwrap();
    //         tokenized_bytes_lock.extend(tokens_bytes);
    //     }
    // });

    // let tokenized_bytes = tokenized_bytes.into_inner().unwrap();

    let mut tokenized_bytes: Vec<Vec<Vec<u8>>> = Vec::new();
    // split all text into tokens
    for item in iterator {
        let item: &PyString = item?.extract()?;
        let text = item.to_str()?;
        if text.is_empty() {
            continue;
        }
        let tokens_bytes = tokenize(text, regex);
        tokenized_bytes.extend(tokens_bytes);
    }

    println!("Done tokenizing");
    let bpe_vocab = build_bpe_vocab(tokenized_bytes, max_token_length, vocab_size);
    let python_dict_out = PyDict::new(py);

    // convert bpe_vocab to python dict
    for (key, value) in bpe_vocab {
        let py_key = PyBytes::new(py, &key);
        python_dict_out.set_item(py_key, value)?;
    }

    Ok(python_dict_out.into())
}

/// bpeasy is a bare-bones implementation of byte-pair encoding (BPE) in Rust.
/// It is designed to be used as a Python module and returns a byte-pair vocabulary
/// as a Python dictionary.
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
        let text = "a b c";
        let regex = r"([^\s]+)|(\s+)";
        let tokens = tokenize(text, regex);
        assert_eq!(
            tokens,
            vec![
                vec![vec![97]],
                vec![vec![32]],
                vec![vec![98]],
                vec![vec![32]],
                vec![vec![99]]
            ]
        );
    }

    #[test]
    fn test_all() {
        let text: &str = "\tYou hear £ £ £ here";
        let regex = r"([^\s]+)|(\s+)";
        let tokenized_bytes = tokenize(text, regex);
        println!("{:?}", tokenized_bytes);

        let vocab_size = 10;
        let max_token_length = 128;
        let bpe_vocab = build_bpe_vocab(tokenized_bytes, max_token_length, vocab_size);
        println!("{:?}", bpe_vocab);
    }

    #[test]
    fn test_initialize_vocab_bytes() {
        let vocab = initialize_vocab_bytes();
        assert_eq!(vocab.len(), 255);
    }
}
