use fancy_regex::Regex;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyIterator, PyString};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Mutex;

#[derive(Debug, Eq)]
struct Merge {
    pair: (Vec<u8>, Vec<u8>),
    count: u128,
    pos: HashSet<u128>,
}
impl PartialEq for Merge {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}
impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Merge {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.count != other.count {
            self.count.cmp(&other.count)
        } else {
            // Here we want ascending order
            other.pair.cmp(&self.pair)
        }
    }
}

type Pair = (Vec<u8>, Vec<u8>);

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
    tokenized_bytes: &[Vec<Vec<u8>>],
    max_token_length: usize,
) -> (HashMap<Pair, u128>, HashMap<Pair, HashSet<u128>>) {
    // Calculate frequencies for each pair of bytes in all sentences and words
    return tokenized_bytes
        .par_iter()
        .enumerate()
        .map(|(i, sentence)| {
            let mut local_pair_counts = HashMap::new();
            let mut local_pair_positions = HashMap::new();
            // let mut local_freqs = HashMap::new();
            for word in sentence.windows(2) {
                if word[0].len() + word[1].len() > max_token_length {
                    continue;
                }
                let current_pair: Pair = (word[0].to_vec(), word[1].to_vec());

                // Initialize pair_counts for this pair if we just saw it for the first time
                local_pair_counts
                    .entry(current_pair.clone())
                    .and_modify(|c| *c += 1)
                    .or_insert(1 as u128);

                // Then update position
                local_pair_positions
                    .entry(current_pair)
                    .and_modify(|h: &mut HashSet<u128>| {
                        h.insert(i as u128);
                    })
                    .or_insert_with(|| {
                        let mut h = HashSet::new();
                        h.insert(i as u128);
                        h
                    });
            }
            (local_pair_counts, local_pair_positions)
        })
        .reduce(
            || (HashMap::new(), HashMap::new()),
            |(mut global_pair_counts, mut global_pair_positions), (pc, wtu)| {
                // Merge the pair counts and positions from all sentences
                for (k, v) in pc {
                    global_pair_counts
                        .entry(k)
                        .and_modify(|c| *c += v)
                        .or_insert(v);
                }
                for (k, v) in wtu {
                    global_pair_positions
                        .entry(k)
                        .and_modify(|set| *set = set.union(&v).copied().collect())
                        .or_insert(v);
                }
                (global_pair_counts, global_pair_positions)
            },
        );
}

fn build_bpe_vocab(
    mut tokenized_bytes: Vec<Vec<Vec<u8>>>,
    max_token_length: usize,
    vocab_size: usize,
) -> HashMap<Vec<u8>, u64> {
    let mut vocab: HashMap<Vec<u8>, u64> = initialize_vocab_bytes();

    let (mut global_pair_counts, mut global_pair_positions) =
        get_most_frequent_pair(&tokenized_bytes, max_token_length);

    // build Priority Queue from counts and positions
    let mut queue: BinaryHeap<Merge> = BinaryHeap::new();
    global_pair_positions.drain().for_each(|(pair, pos)| {
        let count: u128 = global_pair_counts[&pair];
        if count > 0 {
            queue.push(Merge { pair, count, pos });
        }
    });

    let mut num_token_added = 0;
    while num_token_added < vocab_size {
        if queue.is_empty() {
            break;
        }
        let mut top = queue.pop().unwrap();

        if top.count != global_pair_counts[&top.pair] {
            println!("Updating count for {:?}", top.pair);
            top.count = global_pair_counts[&top.pair];
            queue.push(top);
            continue;
        }

        if top.count < 1 {
            break;
        }

        // add to vocab
        let (left, right) = top.pair;
        let merged = [&left[..], &right[..]].concat();
        vocab.insert(merged.clone(), vocab.len() as u64);
        num_token_added += 1;

        let token_str = String::from_utf8_lossy(&merged);
        println!("Token added: {:?}", token_str);
        println!("Pair: {:?}", (left.clone(), right.clone()));
        println!("Count: {}", top.count);

        // update counts and positions
        let mut new_merges: HashSet<Pair> = HashSet::new();
        top.pos.iter().for_each(|&i| {
            // let mut changes = vec![];
            let mut j = 0;
            while j < tokenized_bytes[i as usize].len() - 1 {
                if tokenized_bytes[i as usize][j] == left
                    && tokenized_bytes[i as usize][j + 1] == right
                {
                    println!("Found pair at position: {}", j);
                    // decrement count for old pairs
                    if j > 0 {
                        let prev = tokenized_bytes[i as usize][j - 1].clone();
                        global_pair_counts
                            .entry((prev.clone(), left.clone()))
                            .and_modify(|c| {
                                if *c > 0 {
                                    *c -= 1
                                }
                            })
                            .or_insert(0);
                        // remove i from global_pair_positions
                        global_pair_positions
                            .entry((prev.clone(), left.clone()))
                            .and_modify(|set| {
                                set.remove(&i);
                                if set.is_empty() {
                                    set.remove(&(i as u128));
                                }
                            });
                    }
                    if j < tokenized_bytes[i as usize].len() - 2 {
                        let next = tokenized_bytes[i as usize][j + 2].clone();
                        println!("decrementing : {:?}", (right.clone(), next.clone()));
                        global_pair_counts
                            .entry((right.clone(), next.clone()))
                            .and_modify(|c| {
                                if *c > 0 {
                                    *c -= 1
                                }
                            })
                            .or_insert(0);
                        println!(
                            "count after decrement: {}",
                            global_pair_counts[&(right.clone(), next.clone())]
                        );

                        // remove i from global_pair_positions
                        global_pair_positions
                            .entry((right.clone(), next.clone()))
                            .and_modify(|set| {
                                set.remove(&i);
                                if set.is_empty() {
                                    set.remove(&(i as u128));
                                }
                            });
                    }

                    // Merge the pair and replace the first element with the merged pair
                    let merged = [
                        &tokenized_bytes[i as usize][j][..],
                        &tokenized_bytes[i as usize][j + 1][..],
                    ]
                    .concat();
                    tokenized_bytes[i as usize][j] = merged.clone();

                    // Remove the second element of the pair
                    tokenized_bytes[i as usize].remove(j + 1);

                    // increment count for new pairs
                    if j > 0
                        && merged.len() + tokenized_bytes[i as usize][j - 1].len()
                            < max_token_length
                    {
                        let prev = tokenized_bytes[i as usize][j - 1].clone();

                        global_pair_counts
                            .entry((prev.clone(), merged.clone()))
                            .and_modify(|c| *c += 1)
                            .or_insert(1);
                        global_pair_positions
                            .entry((prev.clone(), merged.clone()))
                            .and_modify(|set| {
                                let new_items: HashSet<_> = vec![i].into_iter().collect();
                                *set = set.union(&new_items).copied().collect();
                            })
                            .or_insert_with(|| {
                                let mut new_set = HashSet::new();
                                new_set.insert(i);
                                new_set
                            });

                        // add to new_merges
                        new_merges.insert((prev.clone(), merged.clone()));
                    }
                    if j < tokenized_bytes[i as usize].len() - 1
                        && merged.len() + tokenized_bytes[i as usize][j + 1].len()
                            < max_token_length
                    {
                        let next = tokenized_bytes[i as usize][j + 1].clone();
                        println!("incrementing : {:?}", (merged.clone(), next.clone()));
                        global_pair_counts
                            .entry((merged.clone(), next.clone()))
                            .and_modify(|c| *c += 1)
                            .or_insert(1);
                        println!(
                            "count after increment: {}",
                            global_pair_counts[&(merged.clone(), next.clone())]
                        );
                        global_pair_positions
                            .entry((merged.clone(), next.clone()))
                            .and_modify(|set| {
                                let new_items: HashSet<_> = vec![i].into_iter().collect();
                                *set = set.union(&new_items).copied().collect();
                            })
                            .or_insert_with(|| {
                                let mut new_set = HashSet::new();
                                new_set.insert(i);
                                new_set
                            });
                        new_merges.insert((merged.clone(), next.clone()));
                    }
                }
                j += 1; // Move to the next token
            }
        });
        // update queue
        new_merges.iter().for_each(|pair| {
            let count = global_pair_counts[pair];
            let pos = global_pair_positions[pair].clone();
            queue.push(Merge {
                pair: pair.clone(),
                count,
                pos,
            });
        });
    }
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

    let tokenized_bytes = Mutex::new(Vec::new());

    // Extract strings from Python iterator and store them in a Rust Vec for parallel processing
    let strings: Vec<&str> = iterator
        .filter_map(|item_result| {
            item_result.ok().and_then(|item| {
                item.extract::<&PyString>()
                    .ok()
                    .and_then(|py_string| py_string.to_str().ok())
            })
        })
        .collect();

    // split all text into tokens
    strings.par_iter().for_each(|text| {
        if !text.is_empty() {
            // println!("Text: {:?}", text);
            let tokens_bytes = tokenize(text, regex);
            // Lock the mutex and extend the vector
            let mut tokenized_bytes_lock = tokenized_bytes.lock().unwrap();
            tokenized_bytes_lock.extend(tokens_bytes);
        }
    });

    let tokenized_bytes = tokenized_bytes.into_inner().unwrap();

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
