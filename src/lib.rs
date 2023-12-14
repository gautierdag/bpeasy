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

    fn from_str(s: &str) -> Self {
        let mut sentence = Sentence::new();
        for byte in s.bytes() {
            sentence.add(byte as u32, 1);
        }
        sentence
    }
}

fn pretokenize<'a>(text: &'a str, regex: &Regex) -> Vec<&'a str> {
    regex
        .find_iter(text)
        .filter_map(|mat| match mat {
            Ok(m) => Some(m.as_str()),
            Err(_) => None,
        })
        .collect()
}

fn pretokenize_strings(strings: Vec<&str>, pattern: &str) -> (Vec<Sentence>, Vec<u64>) {
    let regex: Regex = Regex::new(pattern).expect("Invalid regex pattern");
    let (tokens, counts): (Vec<&str>, Vec<u64>) = strings
        .par_iter()
        .flat_map(|&text| pretokenize(text, &regex))
        .fold(
            || HashMap::new(),
            |mut acc, token| {
                *acc.entry(token).or_insert(0) += 1;
                acc
            },
        )
        .reduce(
            || HashMap::new(),
            |mut a, b| {
                for (token, count) in b {
                    *a.entry(token).or_insert(0) += count;
                }
                a
            },
        )
        .into_iter()
        .unzip();

    let sentences: Vec<Sentence> = tokens.into_iter().map(Sentence::from_str).collect();
    (sentences, counts)
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
    tokenized_sentences: &[Sentence],
    base_counts: &[u64],
) -> (HashMap<Pair, i64>, HashMap<Pair, HashSet<usize>>) {
    // Calculate frequencies for each pair of bytes in all sentences and words
    tokenized_sentences
        .par_iter()
        .enumerate()
        .map(|(i, sentence)| {
            let mut local_pair_counts = HashMap::new();
            let mut local_pair_positions: HashMap<Pair, HashSet<usize>> = HashMap::new();

            for window in sentence.get_symbols().windows(2) {
                let current_pair: Pair = (window[0], window[1]);
                // First update counts
                local_pair_counts
                    .entry(current_pair)
                    .and_modify(|c| *c += base_counts[i] as i64)
                    .or_insert(base_counts[i] as i64);

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
        )
}

// Build vocab from most frequent pairs
fn build_bpe_vocab(
    tokenized_sentences: Vec<Sentence>,
    base_counts: &[u64],
    max_token_length: usize,
    vocab_size: usize,
) -> HashMap<Vec<u8>, u32> {
    let (mut word_to_id, mut id_to_word) = initialize_vocab_bytes(vocab_size);

    // get most frequent pair
    let (mut global_pair_counts, mut global_pair_positions) =
        get_most_frequent_pair(&tokenized_sentences, &base_counts);

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
            // adjust count to reflect sentence level count
            let count = change * base_counts[iw] as i64;
            global_pair_counts
                .entry(pair)
                .and_modify(|c| *c += count)
                .or_insert(count);
            if count > 0 {
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
        .filter(|text| !text.is_empty())
        .collect();

    let (pretokenized_sentences, counts): (Vec<Sentence>, Vec<u64>) =
        pretokenize_strings(strings, regex);

    let bpe_vocab = build_bpe_vocab(
        pretokenized_sentences,
        &counts,
        max_token_length,
        vocab_size,
    );
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
        let pattern = r"([^\s]+)|(\s+)";
        let compiled_regex = fancy_regex::Regex::new(pattern).expect("Invalid regex pattern");
        let pretokenized_sentences = crate::pretokenize(text, &compiled_regex);
        println!("{:?}", pretokenized_sentences);

        let text_2: &str = "You hear £ £ £ here";

        let (pretokenized_sentences, _counts) =
            crate::pretokenize_strings(vec![text, text_2], pattern);

        let vocab_size = 300;
        let max_token_length = 128;
        crate::build_bpe_vocab(
            pretokenized_sentences,
            &_counts,
            max_token_length,
            vocab_size,
        );
    }

    #[test]
    fn test_initialize_vocab_bytes() {
        let vocab = crate::initialize_vocab_bytes(400);
        assert_eq!(vocab.0.len(), 255);
    }
}
