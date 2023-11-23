use fancy_regex::Regex;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyIterator, PyString};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

type Pair = (u32, u32);

#[derive(Debug, Eq)]
struct Merge {
    pair: Pair,
    count: i64,
    pos: HashSet<usize>,
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

#[derive(Debug, Clone, Copy)]
struct Symbol {
    c: u32,
    prev: isize,
    next: isize,
    len: usize,
}

#[derive(Debug)]
struct Sentence {
    symbols: Vec<Symbol>,
}

impl Sentence {
    fn new() -> Self {
        Sentence { symbols: vec![] }
    }

    fn add(&mut self, c: u32, byte_len: usize) {
        let (prev, next) = {
            let len: isize = self.symbols.len() as isize;
            if let Some(last) = self.symbols.last_mut() {
                // Update `next` on the previous one
                last.next = len;
                (len - 1, -1)
            } else {
                (-1, -1)
            }
        };
        self.symbols.push(Symbol {
            c,
            prev,
            next,
            len: byte_len,
        });
    }

    fn merge(&mut self, c1: u32, c2: u32, replacement: u32, max_length: usize) -> Vec<(Pair, i64)> {
        let mut changes: Vec<(Pair, i64)> = vec![];
        let mut i = 0;
        loop {
            if i >= self.symbols.len() {
                break;
            }

            // Found a pair
            if self.symbols[i].c == c1 && i + 1 < self.symbols.len() && self.symbols[i + 1].c == c2
            {
                let first = self.symbols[i];
                let second = self.symbols[i + 1];

                // Remove in place
                let new_s = Symbol {
                    c: replacement,
                    prev: first.prev,
                    next: second.next,
                    len: first.len + second.len,
                };

                // If there are other characters before the pair
                if i > 0 {
                    changes.push(((self.symbols[i - 1].c, first.c), -1));
                    if self.symbols[i - 1].len + new_s.len < max_length {
                        changes.push(((self.symbols[i - 1].c, replacement), 1));
                    }
                }

                self.symbols.insert(i, new_s); // Insert replacement before first char of pair
                self.symbols.remove(i + 1); // Remove first char of pair
                self.symbols.remove(i + 1); // And then the second

                // If there are other characters after the pair
                if i < self.symbols.len() - 1 {
                    changes.push(((second.c, self.symbols[i + 1].c), -1));
                    if self.symbols[i + 1].len + new_s.len < max_length {
                        changes.push(((replacement, self.symbols[i + 1].c), 1));
                    }
                }
            }
            i += 1;
        }
        changes
    }

    fn get_symbols(&self) -> Vec<u32> {
        self.symbols.iter().map(|s| s.c).collect()
    }
}

fn pretokenize(text: &str, pattern: &str) -> Vec<Sentence> {
    let regex = Regex::new(pattern);

    let mut pretokenized_sentences: Vec<Sentence> = Vec::new();

    for match_result in regex.expect(pattern).find_iter(text) {
        match match_result {
            Ok(token) => {
                let mut sentence: Sentence = Sentence::new();
                for byte in token.as_str().bytes() {
                    // tokenized_byte.push(byte as u32);
                    sentence.add(byte as u32, 1);
                }
                pretokenized_sentences.push(sentence);
            }
            Err(e) => {
                println!("Error: {:?}", e);
                break;
            }
        }
    }
    pretokenized_sentences
}

fn initialize_vocab_bytes(vocab_size: usize) -> (HashMap<Vec<u8>, u32>, Vec<Vec<u8>>) {
    let mut word_to_id: HashMap<Vec<u8>, u32> = HashMap::with_capacity(vocab_size);
    let mut id_to_word: Vec<Vec<u8>> = Vec::with_capacity(vocab_size);
    for i in 0..255 {
        word_to_id.insert(vec![i], i as u32);
        id_to_word.push(vec![i]);
    }
    return (word_to_id, id_to_word);
}

fn get_most_frequent_pair(
    tokenized_sentences: &Vec<Sentence>,
) -> (HashMap<Pair, i64>, HashMap<Pair, HashSet<usize>>) {
    // Calculate frequencies for each pair of bytes in all sentences and words
    return tokenized_sentences
        .par_iter()
        .enumerate()
        .map(|(i, sentence)| {
            let mut local_pair_counts = HashMap::new();
            let mut local_pair_positions = HashMap::new();
            for word in sentence.get_symbols().windows(2) {
                let current_pair: Pair = (word[0], word[1]);

                // Initialize pair_counts for this pair if we just saw it for the first time
                local_pair_counts
                    .entry(current_pair)
                    .and_modify(|c| *c += 1)
                    .or_insert(1);

                // Then update position
                local_pair_positions
                    .entry(current_pair)
                    .and_modify(|h: &mut HashSet<usize>| {
                        h.insert(i);
                    })
                    .or_insert_with(|| {
                        let mut h = HashSet::new();
                        h.insert(i);
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

// Build vocab from most frequent pairs
fn build_bpe_vocab(
    tokenized_sentences: Vec<Sentence>,
    max_token_length: usize,
    vocab_size: usize,
) -> HashMap<Vec<u8>, u32> {
    let (mut word_to_id, mut id_to_word) = initialize_vocab_bytes(vocab_size);
    let (mut global_pair_counts, mut global_pair_positions) =
        get_most_frequent_pair(&tokenized_sentences);

    // build Priority Queue from counts and positions
    let mut queue: BinaryHeap<Merge> = BinaryHeap::new();
    global_pair_positions.drain().for_each(|(pair, pos)| {
        let count: i64 = global_pair_counts[&pair];
        if count > 0 {
            queue.push(Merge { pair, count, pos });
        }
    });

    while word_to_id.len() < vocab_size {
        // check if queue is empty
        if queue.is_empty() {
            break;
        }

        let mut top = queue.pop().unwrap();
        // check if count has changed
        if top.count != global_pair_counts[&top.pair] {
            top.count = global_pair_counts[&top.pair];
            queue.push(top);
            continue;
        }

        // exit count is 0
        if top.count < 1 {
            break;
        }

        // add to vocab
        let (left, right) = top.pair;
        let merged_id = word_to_id.len() as u32;

        let mut word = id_to_word[left as usize].clone();
        let right_word = id_to_word[right as usize].clone();
        word.extend(right_word.iter());
        word_to_id.insert(word.clone(), merged_id);
        id_to_word.push(word);

        // update counts and positions for each sentence
        let changes = top
            .pos
            .par_iter()
            .flat_map(|&i| {
                let sentence = &tokenized_sentences[i] as *const _ as *mut Sentence;
                // We can merge each of these sentences in parallel here because each position
                // can be there only once (HashSet). So this is safe.
                unsafe {
                    (*sentence)
                        .merge(top.pair.0, top.pair.1, merged_id, max_token_length)
                        .into_iter()
                        .map(|c| (c, i))
                        .collect::<Vec<_>>()
                }
            })
            .collect::<Vec<_>>();

        for ((pair, change), iw) in changes {
            global_pair_counts
                .entry(pair)
                .and_modify(|c| *c += change)
                .or_insert(change);
            if change > 0 {
                global_pair_positions
                    .entry(pair)
                    .and_modify(|h| {
                        h.insert(iw);
                    })
                    .or_insert_with(|| {
                        let mut h = HashSet::new();
                        h.insert(iw);
                        h
                    });
            }
        }

        // update queue
        global_pair_positions.drain().for_each(|(pair, pos)| {
            let count = global_pair_counts[&pair];
            if count > 0 {
                queue.push(Merge { pair, count, pos });
            }
        });
    }
    word_to_id
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
    if vocab_size < 256 {
        return Err(exceptions::PyValueError::new_err(
            "vocab_size must be greater than 256",
        ));
    }
    if regex.is_empty() {
        return Err(exceptions::PyValueError::new_err("regex cannot be empty"));
    }

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

    let pretokenized_sentences: Vec<Sentence> = strings
        .par_iter()
        .filter(|text| !text.is_empty()) // Filter out empty strings
        .map(|text| pretokenize(text, &regex)) // Tokenize non-empty strings
        .reduce(
            || Vec::new(),
            |mut acc, sentences| {
                acc.extend(sentences);
                acc
            },
        );

    println!("Done tokenizing");
    let bpe_vocab = build_bpe_vocab(pretokenized_sentences, max_token_length, vocab_size);
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
    #[test]
    fn test_all() {
        let text: &str = "\tYou hear £ £ £ here";
        let regex = r"([^\s]+)|(\s+)";
        let pretokenized_sentences = crate::pretokenize(text, regex);
        println!("{:?}", pretokenized_sentences);

        let vocab_size = 300;
        let max_token_length = 128;
        crate::build_bpe_vocab(pretokenized_sentences, max_token_length, vocab_size);
    }

    #[test]
    fn test_initialize_vocab_bytes() {
        let vocab = crate::initialize_vocab_bytes(400);
        assert_eq!(vocab.0.len(), 255);
    }
}
