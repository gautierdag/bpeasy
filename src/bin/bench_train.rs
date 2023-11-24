use serde_json::Value;
use std::f64;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

include!("../lib.rs");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let regex = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    let vocab_size = 1000;
    let max_token_length = 128;

    let file = File::open("./benchmarks/data/c4.jsonl")?;
    let reader = BufReader::new(file);

    // Iterate over each line in the file
    let max_chars = 1_000_000;
    let mut char_count = 0;
    let mut strings: Vec<String> = Vec::new();
    for line in reader.lines() {
        if char_count > max_chars {
            break;
        }
        let line = line.unwrap();

        // Parse the JSON
        let json: Value = serde_json::from_str(&line).expect("Failed to parse JSON");

        // Access the "text" field
        if let Some(text) = json["text"].as_str() {
            char_count += &text.chars().count();
            strings.push(text.to_string());
        }
    }

    println!("Tokenized bytes");
    let mut durations = Vec::new();

    for _ in 0..3 {
        let start = Instant::now();

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

        // Replace with your actual function call and parameters
        crate::build_bpe_vocab(pretokenized_sentences, max_token_length, vocab_size);

        let duration = start.elapsed();
        durations.push(duration.as_secs_f64());
    }

    let avg_duration = durations.iter().sum::<f64>() / durations.len() as f64;
    let variance = durations
        .iter()
        .map(|&duration| {
            let diff = duration - avg_duration;
            diff * diff
        })
        .sum::<f64>()
        / durations.len() as f64;
    let std_deviation = variance.sqrt();

    println!("Average Duration: {} seconds", avg_duration);
    println!("Standard Deviation: {} seconds", std_deviation);

    Ok(())
}
