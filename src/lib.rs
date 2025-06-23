use fancy_regex::Regex;
use fxhash::FxHashMap as HashMap;
use fxhash::FxHashSet as HashSet;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyIterator, PyString};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

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


fn initialize_vocab_bytes(vocab_size: usize) -> (HashMap<Vec<u8>, u32>, Vec<Vec<u8>>) {
    let mut word_to_id: HashMap<Vec<u8>, u32> = HashMap::default();
    let mut id_to_word: Vec<Vec<u8>> = Vec::with_capacity(vocab_size);
    for i in 0..=255 {
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
            let mut local_pair_counts = HashMap::<Pair, i64>::default();
            let mut local_pair_positions: HashMap<Pair, HashSet<usize>> = HashMap::default();

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
                        let mut h = HashSet::<usize>::default();
                        h.insert(i);
                        h
                    });
            }
            (local_pair_counts, local_pair_positions)
        })
        .reduce(
            || {
                (
                    HashMap::<Pair, i64>::default(),
                    HashMap::<Pair, HashSet<usize>>::default(),
                )
            },
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
                        let mut h = HashSet::<usize>::default();
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


// Helper function to process a batch in parallel
fn tokenize_strings(batch: &[String], regex: &Regex) -> HashMap<String, u64> {
    batch
        .par_iter()
        .flat_map(|text| pretokenize(text, regex))
        .fold(
            || HashMap::<String, u64>::default(),
            |mut acc, token| {
                *acc.entry(token.to_string()).or_insert(0) += 1;
                acc
            },
        )
        .reduce(
            || HashMap::<String, u64>::default(),
            |mut a, b| {
                for (token, count) in b {
                    *a.entry(token).or_insert(0) += count;
                }
                a
            },
        )
}

// Helper function to merge token counts
fn merge_token_counts(global: &mut HashMap<String, u64>, batch: HashMap<String, u64>) {
    for (token, count) in batch {
        *global.entry(token).or_insert(0) += count;
    }
}

// Extracted validation function (non-PyO3 version for testing)
fn validate_bpe_parameters_internal(
    max_token_length: usize,
    vocab_size: usize,
    regex: &str,
    batch_size: usize,
) -> Result<(), String> {
    if max_token_length < 2 {
        return Err("max_token_length must be greater than 1".to_string());
    }
    if vocab_size < 256 {
        return Err("vocab_size must be greater than 256".to_string());
    }
    if regex.is_empty() {
        return Err("regex cannot be empty".to_string());
    }
    if batch_size < 1 {
        return Err("batch_size must be greater than 0".to_string());
    }
    Ok(())
}

// PyO3 wrapper for validation
fn validate_bpe_parameters(
    max_token_length: usize,
    vocab_size: usize,
    regex: &str,
    batch_size: usize,
) -> PyResult<()> {
    validate_bpe_parameters_internal(max_token_length, vocab_size, regex, batch_size)
        .map_err(|e| exceptions::PyValueError::new_err(e))
}

// Extracted batch processing function
fn process_iterator_in_batches(
    iterator: Bound<PyIterator>,
    regex: &Regex,
    batch_size: usize,
) -> PyResult<HashMap<String, u64>> {
    let mut token_counts = HashMap::<String, u64>::default();
    let mut batch: Vec<String> = Vec::with_capacity(batch_size);
    
    // Process iterator in batches for parallel processing
    for item_result in iterator {
        if let Ok(item) = item_result {
            if let Ok(py_string) = item.downcast::<PyString>() {
                let text = py_string.to_string_lossy().to_string();
                if !text.is_empty() {
                    batch.push(text);
                    // Process batch when full
                    if batch.len() >= batch_size {
                        let batch_counts = tokenize_strings(&batch, regex);
                        merge_token_counts(&mut token_counts, batch_counts);
                        batch.clear();
                    }
                }
            }
        }
    }
    
    // Process remaining items in batch
    if !batch.is_empty() {
        let batch_counts = tokenize_strings(&batch, regex);
        merge_token_counts(&mut token_counts, batch_counts);
    }
    Ok(token_counts)
}

// Extracted token filtering function
fn filter_tokens_to_sentences(token_counts: HashMap<String, u64>) -> (Vec<Sentence>, Vec<u64>) {
    let (tokens, counts): (Vec<String>, Vec<u64>) = token_counts.into_iter().unzip();
    let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
    
    // Convert tokens to sentences and filter
    token_refs
        .into_iter()
        .map(Sentence::from_str)
        .zip(counts.into_iter())
        .filter(|(sentence, _)| sentence.symbols.len() > 1)
        .unzip()
}

// Refactored main function
#[pyfunction]
fn train_bpe_stream(
    py: Python,
    iterator: Bound<PyIterator>,
    python_regex: Bound<PyString>,
    max_token_length: usize,
    vocab_size: usize,
    batch_size: usize,
) -> PyResult<PyObject> {
    let regex_str = python_regex.to_string_lossy();
    
    // Validate inputs
    validate_bpe_parameters(max_token_length, vocab_size, &regex_str, batch_size)?;
    
    // Compile regex
    let compiled_regex = Regex::new(&regex_str).map_err(|e| 
        exceptions::PyValueError::new_err(format!("Invalid regex: {}", e))
    )?;
    
    // Process iterator and collect token counts
    let token_counts = process_iterator_in_batches(iterator, &compiled_regex, batch_size)?;
    
    // Filter tokens and convert to sentences
    let (filtered_sentences, filtered_counts) = filter_tokens_to_sentences(token_counts);
    
    // Build BPE vocabulary
    let bpe_vocab = build_bpe_vocab(
        filtered_sentences,
        &filtered_counts,
        max_token_length,
        vocab_size,
    );
    
    // Convert to Python dictionary
    let python_dict_out = PyDict::new(py);
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
fn bpeasy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_bpe_stream, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_all() {
        let text: &str = "\tYou hear a £ £ £ here";
        let pattern = r"([^\s]+)|(\s+)";
        let compiled_regex: fancy_regex::Regex =
            fancy_regex::Regex::new(pattern).expect("Invalid regex pattern");
        let pretokenized_text = crate::pretokenize(text, &compiled_regex);
        assert_eq!(
            pretokenized_text,
            vec!["\t", "You", " ", "hear", " ", "a", " ", "£", " ", "£", " ", "£", " ", "here"]
        );

        let text_2: &str = "You hear £ £ £ here";
        let pretokenized_text_2 = crate::pretokenize(text_2, &compiled_regex);

        // Create sentences from pretokenized text
        let mut sentences = Vec::new();
        let mut counts = Vec::new();
        
        for token in pretokenized_text.iter().chain(pretokenized_text_2.iter()) {
            let sentence = crate::Sentence::from_str(token);
            if sentence.symbols.len() > 1 {
                sentences.push(sentence);
                counts.push(1u64);
            }
        }

        let vocab_size = 300;
        let max_token_length = 128;
        crate::build_bpe_vocab(
            sentences,
            &counts,
            max_token_length,
            vocab_size,
        );
    }

    #[test]
    fn test_initialize_vocab_bytes() {
        let vocab = crate::initialize_vocab_bytes(400);
        assert_eq!(vocab.0.len(), 256);
    }
    
    #[test]
    fn test_validate_bpe_parameters() {
        // Valid parameters should pass
        assert!(validate_bpe_parameters_internal(10, 300, "test", 100).is_ok());
        
        // Invalid max_token_length
        assert!(validate_bpe_parameters_internal(1, 300, "test", 100).is_err());
        
        // Invalid vocab_size
        assert!(validate_bpe_parameters_internal(10, 200, "test", 100).is_err());
        
        // Empty regex
        assert!(validate_bpe_parameters_internal(10, 300, "", 100).is_err());
        
        // Invalid batch_size
        assert!(validate_bpe_parameters_internal(10, 300, "test", 0).is_err());
    }
    
    #[test]
    fn test_filter_tokens_to_sentences() {
        let mut token_counts = HashMap::default();
        token_counts.insert("a".to_string(), 5u64);      // Single char - should be filtered
        token_counts.insert("hello".to_string(), 10u64); // Multi char - should be kept
        token_counts.insert("world".to_string(), 3u64);  // Multi char - should be kept
        
        let (sentences, counts) = filter_tokens_to_sentences(token_counts);
        
        assert_eq!(sentences.len(), 2);
        assert_eq!(counts.len(), 2);
        
        // Check that all sentences have more than 1 symbol
        for sentence in &sentences {
            assert!(sentence.symbols.len() > 1);
        }
        
        // Check that counts match (order might vary due to HashMap)
        let total_count: u64 = counts.iter().sum();
        assert_eq!(total_count, 13); // 10 + 3, excluding the single char token
    }
    
    #[test]
    fn test_merge_token_counts() {
        let mut global = HashMap::default();
        global.insert("hello".to_string(), 5u64);
        
        let mut batch = HashMap::default();
        batch.insert("hello".to_string(), 3u64);
        batch.insert("world".to_string(), 2u64);
        
        merge_token_counts(&mut global, batch);
        
        assert_eq!(global.get("hello"), Some(&8u64));
        assert_eq!(global.get("world"), Some(&2u64));
        assert_eq!(global.len(), 2);
    }
}
