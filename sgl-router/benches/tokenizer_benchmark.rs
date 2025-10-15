//! Comprehensive tokenizer benchmark with clean summary output
//! Each test adds a row to the final summary table

use criterion::{black_box, criterion_group, BenchmarkId, Criterion, Throughput};
use sglang_router_rs::tokenizer::{
    cache::{CacheConfig, CachedTokenizer},
    huggingface::HuggingFaceTokenizer,
    sequence::Sequence,
    stop::*,
    stream::DecodeStream,
    traits::*,
};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

// Cache the tokenizer path for the entire benchmark run
static TOKENIZER_PATH: OnceLock<PathBuf> = OnceLock::new();

fn get_tokenizer_path() -> &'static PathBuf {
    TOKENIZER_PATH.get_or_init(|| {
        // Use DeepSeek-R1 which has chat template support
        // Create a synchronous runtime to download the tokenizer
        let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
        let tokenizer_dir = rt.block_on(async {
            sglang_router_rs::tokenizer::hub::download_tokenizer_from_hf("deepseek-ai/DeepSeek-R1")
                .await
                .expect("Failed to download DeepSeek-R1 tokenizer from HuggingFace")
        });

        // The download_tokenizer_from_hf returns the directory containing tokenizer.json
        // We need to construct the full path to tokenizer.json
        tokenizer_dir.join("tokenizer.json")
    })
}

// Production target: 100k tokens per second
const TARGET_TOKENS_PER_SECOND: u64 = 100_000;

// Typical prompt sizes
const SHORT_PROMPT: &str = "What is the capital of France?";
const MEDIUM_PROMPT: &str = "Write a detailed explanation of quantum computing, including its principles, current applications, and future potential. Be sure to cover both the theoretical foundations and practical implementations.";
const LONG_PROMPT: &str = "You are an expert software engineer. Review the following code and provide detailed feedback on performance optimizations, potential bugs, and architectural improvements. Consider scalability, maintainability, and best practices. The code implements a distributed caching system with the following requirements: 1) High availability across multiple regions, 2) Sub-millisecond latency for cache hits, 3) Automatic failover and recovery, 4) Support for both LRU and LFU eviction policies, 5) Real-time monitoring and alerting. Please analyze each component thoroughly and suggest concrete improvements with code examples where appropriate.";

// System prompts can be quite large
fn generate_system_prompt(size: usize) -> String {
    let base = "You are a helpful AI assistant with expertise in ";
    let domains = vec![
        "mathematics",
        "physics",
        "chemistry",
        "biology",
        "computer science",
        "engineering",
        "medicine",
        "law",
        "economics",
        "philosophy",
    ];

    let mut prompt = base.to_string();
    while prompt.len() < size {
        for domain in &domains {
            prompt.push_str(domain);
            prompt.push_str(", ");
            if prompt.len() >= size {
                break;
            }
        }
    }
    prompt
}

// Global results storage
lazy_static::lazy_static! {
    static ref BENCHMARK_RESULTS: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());
}

fn add_result(category: &str, result: String) {
    let mut results = BENCHMARK_RESULTS.lock().unwrap();
    let index = results.len();
    results.insert(format!("{:03}_{}", index, category), result);
}

fn bench_encode_throughput(c: &mut Criterion) {
    let tokenizer_path = get_tokenizer_path();
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    // Pre-generate system prompts
    let system_1k = generate_system_prompt(1000);
    let system_4k = generate_system_prompt(4000);
    let system_16k = generate_system_prompt(16000);

    let test_cases = vec![
        ("short_30B", SHORT_PROMPT),
        ("medium_230B", MEDIUM_PROMPT),
        ("long_670B", LONG_PROMPT),
        ("system_1KB", system_1k.as_str()),
        ("system_4KB", system_4k.as_str()),
        ("system_16KB", system_16k.as_str()),
    ];

    let mut group = c.benchmark_group("encode_throughput");

    for (name, prompt) in test_cases {
        let prompt_len = prompt.len();
        let tokenizer_clone = tokenizer.clone();

        // Get token count once
        let encoding = tokenizer.encode(prompt).unwrap();
        let token_count = encoding.token_ids().len();

        // Track if metrics have been printed for this test case
        let printed = Arc::new(AtomicBool::new(false));

        group.throughput(Throughput::Bytes(prompt_len as u64));
        group.bench_function(name, |b| {
            let printed_clone = printed.clone();
            let tokenizer = tokenizer_clone.clone();

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    black_box(tokenizer.encode(prompt).unwrap());
                }
                let duration = start.elapsed();

                // Store result only once per test case
                if !printed_clone.load(Ordering::Relaxed) {
                    let ops_per_sec = iters as f64 / duration.as_secs_f64();
                    let chars_per_sec = (iters as f64 * prompt_len as f64) / duration.as_secs_f64();
                    let tokens_per_sec =
                        (iters as f64 * token_count as f64) / duration.as_secs_f64();

                    let result = format!(
                        "{:<15} | {:>8} | {:>8} | {:>12.0} | {:>12.0} | {:>10.0} | {:>10}",
                        name,
                        prompt_len,
                        token_count,
                        chars_per_sec,
                        tokens_per_sec,
                        ops_per_sec,
                        1
                    );
                    add_result("encode", result);

                    printed_clone.store(true, Ordering::Relaxed);
                }

                duration
            });
        });
    }

    group.finish();
}

fn bench_batch_encode(c: &mut Criterion) {
    let tokenizer_path = get_tokenizer_path();
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    let batch_sizes = vec![1, 8, 16, 32, 64, 128];
    let prompt = MEDIUM_PROMPT;
    let prompt_len = prompt.len();
    let encoding = tokenizer.encode(prompt).unwrap();
    let token_count = encoding.token_ids().len();

    let mut group = c.benchmark_group("batch_encode");

    for batch_size in batch_sizes {
        let prompts: Vec<&str> = vec![prompt; batch_size];
        let printed = Arc::new(AtomicBool::new(false));
        let tokenizer_clone = tokenizer.clone();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &size| {
                let printed_clone = printed.clone();
                let tokenizer = tokenizer_clone.clone();

                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        black_box(tokenizer.encode_batch(&prompts).unwrap());
                    }
                    let duration = start.elapsed();

                    if !printed_clone.load(Ordering::Relaxed) {
                        let prompts_per_sec = (iters as f64 * size as f64) / duration.as_secs_f64();
                        let tokens_per_sec = prompts_per_sec * token_count as f64;
                        let chars_per_sec = prompts_per_sec * prompt_len as f64;

                        let result = format!(
                            "{:<15} | {:>8} | {:>8} | {:>12.0} | {:>12.0} | {:>10.0} | {:>10}",
                            format!("batch_{}", size),
                            prompt_len * size,
                            token_count * size,
                            prompts_per_sec,
                            tokens_per_sec,
                            chars_per_sec,
                            1
                        );
                        add_result("batch", result);

                        printed_clone.store(true, Ordering::Relaxed);
                    }

                    duration
                });
            },
        );
    }

    group.finish();
}

fn bench_concurrent_encode(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(get_tokenizer_path().to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    let client_counts = vec![1, 4, 8, 16, 32];

    let mut group = c.benchmark_group("concurrent_encode");
    group.measurement_time(Duration::from_secs(2));

    for num_clients in client_counts {
        let printed = Arc::new(AtomicBool::new(false));
        let tokenizer_clone = tokenizer.clone();

        group.bench_with_input(
            BenchmarkId::from_parameter(num_clients),
            &num_clients,
            |b, &clients| {
                let printed_clone = printed.clone();

                b.iter_custom(|_iters| {
                    let tokenizer = tokenizer_clone.clone();
                    let total_operations = Arc::new(AtomicU64::new(0));
                    let total_chars = Arc::new(AtomicU64::new(0));
                    let start = Instant::now();

                    let handles: Vec<_> = (0..clients)
                        .map(|client_id| {
                            let tokenizer = tokenizer.clone();
                            let total_ops = total_operations.clone();
                            let total_ch = total_chars.clone();

                            thread::spawn(move || {
                                let prompts = [SHORT_PROMPT, MEDIUM_PROMPT, LONG_PROMPT];
                                let prompt = prompts[client_id % prompts.len()];
                                let mut local_ops = 0u64;
                                let mut local_chars = 0u64;

                                while start.elapsed() < Duration::from_millis(500) {
                                    let _ = tokenizer.encode(prompt).unwrap();
                                    local_ops += 1;
                                    local_chars += prompt.len() as u64;
                                }

                                total_ops.fetch_add(local_ops, Ordering::Relaxed);
                                total_ch.fetch_add(local_chars, Ordering::Relaxed);
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }

                    let duration = start.elapsed();

                    if !printed_clone.load(Ordering::Relaxed) {
                        let total_ops = total_operations.load(Ordering::Relaxed);
                        let total_ch = total_chars.load(Ordering::Relaxed);
                        let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
                        let chars_per_sec = total_ch as f64 / duration.as_secs_f64();
                        let per_client = ops_per_sec / clients as f64;

                        let result = format!(
                            "{:<15} | {:>10} | {:>12.0} | {:>12.0} | {:>15.0}",
                            format!("{}_clients", clients),
                            total_ops,
                            ops_per_sec,
                            chars_per_sec,
                            per_client
                        );
                        add_result("concurrent", result);

                        printed_clone.store(true, Ordering::Relaxed);
                    }

                    duration
                });
            },
        );
    }

    group.finish();
}

fn bench_decode_performance(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(get_tokenizer_path().to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    let test_text = "The quick brown fox jumps over the lazy dog. ".repeat(10);
    let encoding = tokenizer.encode(&test_text).unwrap();
    let tokens = encoding.token_ids();
    let num_tokens = tokens.len();

    let mut group = c.benchmark_group("decode_performance");

    // Test direct decode
    let printed_direct = Arc::new(AtomicBool::new(false));
    group.bench_function("direct_decode", |b| {
        let printed = printed_direct.clone();
        let tokenizer = tokenizer.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                black_box(tokenizer.decode(tokens, false).unwrap());
            }
            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let tokens_per_sec = ops_per_sec * num_tokens as f64;

                let result = format!(
                    "{:<20} | {:>10} | {:>12.0} | {:>12.0} | {:>10}",
                    "Direct", num_tokens, tokens_per_sec, ops_per_sec, 1
                );
                add_result("decode", result);

                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    // Test DecodeStream
    let printed_stream = Arc::new(AtomicBool::new(false));
    group.bench_function("decode_stream", |b| {
        let printed = printed_stream.clone();
        let tokenizer = tokenizer.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let mut decoder = DecodeStream::new(tokenizer.clone(), &[], false);
                let mut output = String::new();
                for token in tokens {
                    if let Some(text) = decoder.step(*token).unwrap() {
                        output.push_str(&text);
                    }
                }
                black_box(output);
            }
            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let tokens_per_sec = ops_per_sec * num_tokens as f64;

                let result = format!(
                    "{:<20} | {:>10} | {:>12.0} | {:>12.0} | {:>10}",
                    "DecodeStream", num_tokens, tokens_per_sec, ops_per_sec, 1
                );
                add_result("decode", result);

                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    // Test Sequence
    let printed_seq = Arc::new(AtomicBool::new(false));
    group.bench_function("sequence_decode", |b| {
        let printed = printed_seq.clone();
        let tokenizer = tokenizer.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let mut sequence = Sequence::new(tokenizer.clone());
                let mut output = String::new();
                for token in tokens {
                    let text = sequence.append_token(*token).unwrap();
                    output.push_str(&text);
                }
                black_box(output);
            }
            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let tokens_per_sec = ops_per_sec * num_tokens as f64;

                let result = format!(
                    "{:<20} | {:>10} | {:>12.0} | {:>12.0} | {:>10}",
                    "Sequence", num_tokens, tokens_per_sec, ops_per_sec, 1
                );
                add_result("decode", result);

                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    group.finish();
}

fn bench_streaming_decode_100k(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(get_tokenizer_path().to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    let sample_text = "The quick brown fox jumps over the lazy dog. ".repeat(1000);
    let encoding = tokenizer.encode(&sample_text).unwrap();
    let all_tokens = encoding.token_ids();

    let mut group = c.benchmark_group("streaming_100k");
    group.measurement_time(Duration::from_secs(1));

    // Test DecodeStream
    let printed_stream = Arc::new(AtomicBool::new(false));
    group.bench_function("decode_stream_100k", |b| {
        let printed = printed_stream.clone();
        let tokenizer = tokenizer.clone();

        b.iter_custom(|_iters| {
            let start = Instant::now();
            let mut decoder = DecodeStream::new(tokenizer.clone(), &[], false);
            let mut output = String::new();
            let mut tokens_processed = 0u64;

            for token in all_tokens.iter().cycle() {
                if start.elapsed() >= Duration::from_millis(500) {
                    break;
                }

                if let Some(text) = decoder.step(*token).unwrap() {
                    output.push_str(&text);
                }
                tokens_processed += 1;
            }

            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let tokens_per_sec = tokens_processed as f64 / duration.as_secs_f64();
                let status = if tokens_per_sec >= TARGET_TOKENS_PER_SECOND as f64 {
                    "PASS"
                } else {
                    "BELOW"
                };

                let result = format!(
                    "{:<20} | {:>12} | {:>12.0} | {:>12} | {:>10} | {:>12}",
                    "DecodeStream",
                    tokens_processed,
                    tokens_per_sec,
                    TARGET_TOKENS_PER_SECOND,
                    1,
                    status
                );
                add_result("streaming_100k", result);

                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    // Test Sequence
    let printed_seq = Arc::new(AtomicBool::new(false));
    group.bench_function("sequence_100k", |b| {
        let printed = printed_seq.clone();
        let tokenizer = tokenizer.clone();

        b.iter_custom(|_iters| {
            let start = Instant::now();
            let mut sequence = Sequence::new(tokenizer.clone());
            let mut output = String::new();
            let mut tokens_processed = 0u64;

            for token in all_tokens.iter().cycle() {
                if start.elapsed() >= Duration::from_millis(500) {
                    break;
                }

                let text = sequence.append_token(*token).unwrap();
                output.push_str(&text);
                tokens_processed += 1;
            }

            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let tokens_per_sec = tokens_processed as f64 / duration.as_secs_f64();
                let status = if tokens_per_sec >= TARGET_TOKENS_PER_SECOND as f64 {
                    "PASS"
                } else {
                    "BELOW"
                };

                let result = format!(
                    "{:<20} | {:>12} | {:>12.0} | {:>12} | {:>10} | {:>12}",
                    "Sequence",
                    tokens_processed,
                    tokens_per_sec,
                    TARGET_TOKENS_PER_SECOND,
                    1,
                    status
                );
                add_result("streaming_100k", result);

                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    group.finish();
}

fn bench_latency_distribution(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(get_tokenizer_path().to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    // Test latency for individual token processing
    let sample_tokens = vec![1, 450, 6635, 3290, 491, 278, 3474, 29892];

    let mut group = c.benchmark_group("latency");

    // Encode latency
    let system_4k = generate_system_prompt(4000);
    let test_cases = vec![
        ("encode_short", SHORT_PROMPT),
        ("encode_medium", MEDIUM_PROMPT),
        ("encode_long", LONG_PROMPT),
        ("encode_4KB", system_4k.as_str()),
    ];

    for (name, prompt) in test_cases {
        let printed = Arc::new(AtomicBool::new(false));
        group.bench_function(name, |b| {
            let printed_clone = printed.clone();
            let tokenizer = tokenizer.clone();

            b.iter_custom(|iters| {
                // Only collect detailed latency on first iteration
                let total_duration = if !printed_clone.load(Ordering::Relaxed) {
                    let mut latencies = Vec::new();

                    // Warm up
                    for _ in 0..100 {
                        let _ = tokenizer.encode(prompt).unwrap();
                    }

                    // Measure for statistics
                    for _ in 0..1000 {
                        let start = Instant::now();
                        let _ = tokenizer.encode(prompt).unwrap();
                        let latency = start.elapsed();
                        latencies.push(latency);
                    }

                    latencies.sort();
                    let p50 = latencies[latencies.len() / 2];
                    let p95 = latencies[latencies.len() * 95 / 100];
                    let p99 = latencies[latencies.len() * 99 / 100];
                    let max = latencies.last().unwrap();

                    let result = format!(
                        "{:<20} | {:>10.1} | {:>10.1} | {:>10.1} | {:>10.1} | {:>10}",
                        name,
                        p50.as_micros() as f64,
                        p95.as_micros() as f64,
                        p99.as_micros() as f64,
                        max.as_micros() as f64,
                        1000
                    );
                    add_result("latency", result);

                    printed_clone.store(true, Ordering::Relaxed);

                    // Return median for consistency
                    p50 * iters as u32
                } else {
                    // Regular benchmark iterations
                    let start = Instant::now();
                    for _ in 0..iters {
                        black_box(tokenizer.encode(prompt).unwrap());
                    }
                    start.elapsed()
                };

                total_duration
            });
        });
    }

    // Decode token latency
    let printed_decode = Arc::new(AtomicBool::new(false));
    group.bench_function("decode_token", |b| {
        let printed_clone = printed_decode.clone();
        let tokenizer = tokenizer.clone();
        let tokens = sample_tokens.clone();

        b.iter_custom(|iters| {
            let total_duration = if !printed_clone.load(Ordering::Relaxed) {
                let mut latencies = Vec::new();
                let mut decoder = DecodeStream::new(tokenizer.clone(), &[], false);

                for token in tokens.iter().cycle().take(1000) {
                    let start = Instant::now();
                    let _ = decoder.step(*token).unwrap();
                    let latency = start.elapsed();
                    latencies.push(latency);
                }

                latencies.sort();
                let p50 = latencies[latencies.len() / 2];
                let p95 = latencies[latencies.len() * 95 / 100];
                let p99 = latencies[latencies.len() * 99 / 100];
                let max = latencies.last().unwrap();

                let result = format!(
                    "{:<20} | {:>10.1} | {:>10.1} | {:>10.1} | {:>10.1} | {:>10}",
                    "decode_token",
                    p50.as_micros() as f64,
                    p95.as_micros() as f64,
                    p99.as_micros() as f64,
                    max.as_micros() as f64,
                    1000
                );
                add_result("latency", result);

                // Check target latency
                let target_latency = Duration::from_micros(10);
                if p50 > target_latency {
                    let warning = format!(
                        "WARNING: P50 latency exceeds target of {:?} for 100k tokens/sec",
                        target_latency
                    );
                    add_result("latency_warning", warning);
                }

                printed_clone.store(true, Ordering::Relaxed);

                // Return approximate time for consistency
                p50 * iters as u32
            } else {
                // Regular benchmark iterations
                let start = Instant::now();
                for _ in 0..iters {
                    let mut decoder = DecodeStream::new(tokenizer.clone(), &[], false);
                    for token in tokens.iter().take(10) {
                        black_box(decoder.step(*token).unwrap());
                    }
                }
                start.elapsed()
            };

            total_duration
        });
    });

    group.finish();
}

fn bench_concurrent_streaming(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(get_tokenizer_path().to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    let num_sequences = 16;
    let tokens_per_sequence = 10_000;

    let sample_text = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    let encoding = tokenizer.encode(&sample_text).unwrap();
    let token_batch: Vec<u32> = encoding.token_ids().to_vec();

    let mut group = c.benchmark_group("concurrent_streaming");
    group.measurement_time(Duration::from_secs(2));

    let printed = Arc::new(AtomicBool::new(false));
    group.bench_function("concurrent_16_sequences", |b| {
        let printed_clone = printed.clone();
        let tokenizer = tokenizer.clone();
        let tokens = token_batch.clone();

        b.iter_custom(|_iters| {
            let total_tokens = Arc::new(AtomicU64::new(0));
            let start = Instant::now();

            let handles: Vec<_> = (0..num_sequences)
                .map(|_seq_id| {
                    let tokenizer = tokenizer.clone();
                    let tokens = tokens.clone();
                    let total_tokens = total_tokens.clone();

                    thread::spawn(move || {
                        let mut decoder = DecodeStream::new(tokenizer, &[], false);
                        let mut output = String::new();
                        let mut local_count = 0u64;

                        for token in tokens.iter().cycle().take(tokens_per_sequence) {
                            if let Some(text) = decoder.step(*token).unwrap() {
                                output.push_str(&text);
                            }
                            local_count += 1;
                        }

                        total_tokens.fetch_add(local_count, Ordering::Relaxed);
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }

            let duration = start.elapsed();

            if !printed_clone.load(Ordering::Relaxed) {
                let total = total_tokens.load(Ordering::Relaxed);
                let throughput = total as f64 / duration.as_secs_f64();
                let per_seq = throughput / num_sequences as f64;

                let result = format!(
                    "{:<20} | {:>10} | {:>12.0} | {:>15.0} | {:>15}",
                    format!("{}_sequences", num_sequences),
                    total,
                    throughput,
                    per_seq,
                    num_sequences
                );
                add_result("concurrent_streaming", result);

                printed_clone.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    group.finish();
}

fn bench_stop_sequences(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(get_tokenizer_path().to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    let config = StopSequenceConfig::default()
        .with_stop_sequence("</s>")
        .with_stop_sequence("\n\n")
        .with_stop_sequence("###")
        .with_stop_token(2);

    let sample_text = "Hello world! This is a test. ### Stop here. Continue after.".repeat(100);
    let encoding = tokenizer.encode(&sample_text).unwrap();
    let tokens = encoding.token_ids();

    let mut group = c.benchmark_group("stop_sequences");

    // No stops
    let printed_no_stop = Arc::new(AtomicBool::new(false));
    group.bench_function("no_stops", |b| {
        let printed_clone = printed_no_stop.clone();
        let tokenizer = tokenizer.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            let mut total_tokens = 0u64;

            for _ in 0..iters {
                let mut decoder = StopSequenceDecoder::new(
                    tokenizer.clone(),
                    StopSequenceConfig::default(),
                    false,
                );
                for token in tokens {
                    let _ = decoder.process_token(*token).unwrap();
                    total_tokens += 1;
                }
            }

            let duration = start.elapsed();

            if !printed_clone.load(Ordering::Relaxed) {
                let tokens_per_sec = total_tokens as f64 / duration.as_secs_f64();
                let seq_per_sec = iters as f64 / duration.as_secs_f64();

                let result = format!(
                    "{:<20} | {:>10} | {:>12} | {:>12.0} | {:>10.0}",
                    "No stops", iters, total_tokens, tokens_per_sec, seq_per_sec
                );
                add_result("stop_sequences", result);

                printed_clone.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    // With stops
    let printed_with_stops = Arc::new(AtomicBool::new(false));
    group.bench_function("with_stops", |b| {
        let printed_clone = printed_with_stops.clone();
        let tokenizer = tokenizer.clone();
        let config = config.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            let mut total_tokens = 0u64;
            let mut total_sequences = 0u64;

            for _ in 0..iters {
                let mut decoder =
                    StopSequenceDecoder::new(tokenizer.clone(), config.clone(), false);
                let mut sequence_tokens = 0u64;

                for token in tokens {
                    let result = decoder.process_token(*token).unwrap();
                    sequence_tokens += 1;

                    if matches!(
                        result,
                        SequenceDecoderOutput::Stopped | SequenceDecoderOutput::StoppedWithText(_)
                    ) {
                        break;
                    }
                }

                total_tokens += sequence_tokens;
                total_sequences += 1;
            }

            let duration = start.elapsed();

            if !printed_clone.load(Ordering::Relaxed) {
                let tokens_per_sec = total_tokens as f64 / duration.as_secs_f64();
                let seq_per_sec = total_sequences as f64 / duration.as_secs_f64();

                let result = format!(
                    "{:<20} | {:>10} | {:>12} | {:>12.0} | {:>10.0}",
                    "With stops", total_sequences, total_tokens, tokens_per_sec, seq_per_sec
                );
                add_result("stop_sequences", result);

                printed_clone.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    group.finish();
}

fn bench_multithreaded_encode(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(get_tokenizer_path().to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    let thread_counts = vec![1, 2, 4, 8, 16];
    let operations_per_thread = 1000;

    // Test with medium-sized prompt for balanced workload
    let test_prompt = MEDIUM_PROMPT;

    let mut group = c.benchmark_group("multithreaded_encode");
    group.measurement_time(Duration::from_secs(2));

    let mut baseline_throughput = 0.0;

    for num_threads in thread_counts {
        let printed = Arc::new(AtomicBool::new(false));
        let tokenizer_clone = tokenizer.clone();

        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            &num_threads,
            |b, &threads| {
                let printed_clone = printed.clone();
                let tokenizer = tokenizer_clone.clone();

                b.iter_custom(|_iters| {
                    let total_operations = Arc::new(AtomicU64::new(0));
                    let total_tokens = Arc::new(AtomicU64::new(0));
                    let start = Instant::now();

                    let handles: Vec<_> = (0..threads)
                        .map(|_| {
                            let tokenizer = tokenizer.clone();
                            let total_ops = total_operations.clone();
                            let total_tok = total_tokens.clone();

                            thread::spawn(move || {
                                for _ in 0..operations_per_thread {
                                    let encoding = tokenizer.encode(test_prompt).unwrap();
                                    total_tok.fetch_add(
                                        encoding.token_ids().len() as u64,
                                        Ordering::Relaxed,
                                    );
                                }
                                total_ops
                                    .fetch_add(operations_per_thread as u64, Ordering::Relaxed);
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }

                    let duration = start.elapsed();

                    if !printed_clone.load(Ordering::Relaxed) {
                        let total_ops = total_operations.load(Ordering::Relaxed);
                        let total_tok = total_tokens.load(Ordering::Relaxed);
                        let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
                        let tokens_per_sec = total_tok as f64 / duration.as_secs_f64();

                        if threads == 1 {
                            baseline_throughput = tokens_per_sec;
                        }

                        let efficiency = if threads == 1 {
                            100.0
                        } else {
                            (tokens_per_sec / (baseline_throughput * threads as f64)) * 100.0
                        };

                        let result = format!(
                            "{:<20} | {:>10} | {:>12.0} | {:>12.0} | {:>10} | {:>11.1}%",
                            format!("encode_{}_threads", threads),
                            total_ops,
                            ops_per_sec,
                            tokens_per_sec,
                            threads,
                            efficiency
                        );
                        add_result("mt_encode", result);

                        printed_clone.store(true, Ordering::Relaxed);
                    }

                    duration
                });
            },
        );
    }

    group.finish();
}

fn bench_multithreaded_decode(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(get_tokenizer_path().to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    let thread_counts = vec![1, 2, 4, 8, 16];
    let tokens_per_thread = 5000;

    // Generate tokens for decoding
    let test_text = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    let encoding = tokenizer.encode(&test_text).unwrap();
    let test_tokens: Vec<u32> = encoding.token_ids().to_vec();

    let mut group = c.benchmark_group("multithreaded_decode");
    group.measurement_time(Duration::from_secs(2));

    let mut baseline_throughput = 0.0;

    for num_threads in thread_counts {
        let printed = Arc::new(AtomicBool::new(false));
        let tokenizer_clone = tokenizer.clone();
        let tokens = test_tokens.clone();

        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            &num_threads,
            |b, &threads| {
                let printed_clone = printed.clone();
                let tokenizer = tokenizer_clone.clone();
                let tokens = tokens.clone();

                b.iter_custom(|_iters| {
                    let total_tokens = Arc::new(AtomicU64::new(0));
                    let start = Instant::now();

                    let handles: Vec<_> = (0..threads)
                        .map(|_| {
                            let tokenizer = tokenizer.clone();
                            let tokens = tokens.clone();
                            let total_tok = total_tokens.clone();

                            thread::spawn(move || {
                                let mut decoder = DecodeStream::new(tokenizer, &[], false);
                                let mut output = String::new();
                                let mut local_tokens = 0u64;

                                for token in tokens.iter().cycle().take(tokens_per_thread) {
                                    if let Some(text) = decoder.step(*token).unwrap() {
                                        output.push_str(&text);
                                    }
                                    local_tokens += 1;
                                }

                                total_tok.fetch_add(local_tokens, Ordering::Relaxed);
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }

                    let duration = start.elapsed();

                    if !printed_clone.load(Ordering::Relaxed) {
                        let total = total_tokens.load(Ordering::Relaxed);
                        let tokens_per_sec = total as f64 / duration.as_secs_f64();

                        if threads == 1 {
                            baseline_throughput = tokens_per_sec;
                        }

                        let efficiency = if threads == 1 {
                            100.0
                        } else {
                            (tokens_per_sec / (baseline_throughput * threads as f64)) * 100.0
                        };

                        let result = format!(
                            "{:<20} | {:>12} | {:>12.0} | {:>10} | {:>11.1}%",
                            format!("decode_{}_threads", threads),
                            total,
                            tokens_per_sec,
                            threads,
                            efficiency
                        );
                        add_result("mt_decode", result);

                        printed_clone.store(true, Ordering::Relaxed);
                    }

                    duration
                });
            },
        );
    }

    group.finish();
}

fn bench_memory_efficiency(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(get_tokenizer_path().to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    let large_text = "The quick brown fox jumps over the lazy dog. ".repeat(1000);
    let encoding = tokenizer.encode(&large_text).unwrap();

    let mut group = c.benchmark_group("memory");

    // Track owned baseline time
    let mut owned_time_ns = 0.0;

    // Owned
    let printed_owned = Arc::new(AtomicBool::new(false));
    group.bench_function("token_ids_owned", |b| {
        let printed_clone = printed_owned.clone();
        let encoding = encoding.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let _ = black_box(encoding.token_ids());
            }
            let duration = start.elapsed();

            if !printed_clone.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let time_per_call = duration.as_nanos() as f64 / iters as f64;
                owned_time_ns = time_per_call;

                let result = format!(
                    "{:<20} | {:>12.0} | {:>11.0}ns | {:>12}",
                    "token_ids(owned)", ops_per_sec, time_per_call, "baseline"
                );
                add_result("memory", result);

                printed_clone.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    // Reference
    let printed_ref = Arc::new(AtomicBool::new(false));

    group.bench_function("token_ids_ref", |b| {
        let printed_clone = printed_ref.clone();
        let encoding = encoding.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let _ = black_box(encoding.token_ids());
            }
            let duration = start.elapsed();

            if !printed_clone.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let time_per_call = duration.as_nanos() as f64 / iters as f64;

                // Calculate improvement
                let improvement = if owned_time_ns > 0.0 {
                    format!("{:.1}x faster", owned_time_ns / time_per_call)
                } else {
                    "N/A".to_string()
                };

                let result = format!(
                    "{:<20} | {:>12.0} | {:>11.0}ns | {:>12}",
                    "token_ids_ref(ref)", ops_per_sec, time_per_call, improvement
                );
                add_result("memory", result);

                printed_clone.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    group.finish();
}

fn bench_scaling_characteristics(c: &mut Criterion) {
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(get_tokenizer_path().to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    let thread_counts = vec![1, 2, 4, 8, 16];
    let tokens_per_thread = 10000;

    let mut group = c.benchmark_group("scaling");
    group.measurement_time(Duration::from_secs(2));

    let mut baseline_throughput = 0.0;

    for num_threads in thread_counts {
        let printed = Arc::new(AtomicBool::new(false));

        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            &num_threads,
            |b, &threads| {
                let printed_clone = printed.clone();
                let tokenizer = tokenizer.clone();

                b.iter_custom(|_iters| {
                    let total_tokens = Arc::new(AtomicU64::new(0));
                    let start = Instant::now();

                    let handles: Vec<_> = (0..threads)
                        .map(|_| {
                            let tokenizer = tokenizer.clone();
                            let total_tokens = total_tokens.clone();

                            thread::spawn(move || {
                                let mut decoder = DecodeStream::new(tokenizer, &[], false);
                                let sample_tokens = [1, 450, 6635, 3290, 491];

                                for token in sample_tokens.iter().cycle().take(tokens_per_thread) {
                                    let _ = decoder.step(*token).unwrap();
                                }

                                total_tokens.fetch_add(tokens_per_thread as u64, Ordering::Relaxed);
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }

                    let duration = start.elapsed();

                    if !printed_clone.load(Ordering::Relaxed) {
                        let total = total_tokens.load(Ordering::Relaxed);
                        let throughput = total as f64 / duration.as_secs_f64();

                        if threads == 1 {
                            baseline_throughput = throughput;
                        }

                        let efficiency = if threads == 1 {
                            100.0
                        } else {
                            (throughput / (baseline_throughput * threads as f64)) * 100.0
                        };

                        let result = format!(
                            "{:<15} | {:>12} | {:>12.0} | {:>11.1}%",
                            format!("{}_threads", threads),
                            total,
                            throughput,
                            efficiency
                        );
                        add_result("scaling", result);

                        printed_clone.store(true, Ordering::Relaxed);
                    }

                    duration
                });
            },
        );
    }

    group.finish();
}

fn bench_cached_vs_uncached(c: &mut Criterion) {
    let tokenizer_path = get_tokenizer_path();
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    // Create cached tokenizer
    let cached_tokenizer = Arc::new(CachedTokenizer::new(
        tokenizer.clone(),
        CacheConfig::default(),
    ));

    // Pre-generate prompts
    let system_4k = generate_system_prompt(4000);

    let test_cases = vec![
        ("short_30B", SHORT_PROMPT),
        ("medium_230B", MEDIUM_PROMPT),
        ("long_670B", LONG_PROMPT),
        ("system_4KB", system_4k.as_str()),
    ];

    let mut group = c.benchmark_group("cache_comparison");

    for (name, prompt) in test_cases {
        let prompt_len = prompt.len();

        // Benchmark uncached
        let printed_uncached = Arc::new(AtomicBool::new(false));
        group.bench_function(format!("{}_uncached", name), |b| {
            let printed = printed_uncached.clone();
            let tokenizer = tokenizer.clone();

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    black_box(tokenizer.encode(prompt).unwrap());
                }
                let duration = start.elapsed();

                if !printed.load(Ordering::Relaxed) {
                    let ops_per_sec = iters as f64 / duration.as_secs_f64();
                    let time_per_op = duration.as_micros() as f64 / iters as f64;

                    let result = format!(
                        "{:<20} | {:>8} | {:>12.0} | {:>12.1} | {:>12}",
                        format!("{}_uncached", name),
                        prompt_len,
                        ops_per_sec,
                        time_per_op,
                        "baseline"
                    );
                    add_result("cache", result);

                    printed.store(true, Ordering::Relaxed);
                }

                duration
            });
        });

        // Benchmark cached (first call - miss)
        let printed_miss = Arc::new(AtomicBool::new(false));
        group.bench_function(format!("{}_cache_miss", name), |b| {
            let printed = printed_miss.clone();

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    // Create fresh cached tokenizer for each iteration to ensure miss
                    let fresh_cached =
                        CachedTokenizer::new(tokenizer.clone(), CacheConfig::default());
                    black_box(fresh_cached.encode(prompt).unwrap());
                }
                let duration = start.elapsed();

                if !printed.load(Ordering::Relaxed) {
                    let ops_per_sec = iters as f64 / duration.as_secs_f64();
                    let time_per_op = duration.as_micros() as f64 / iters as f64;

                    let result = format!(
                        "{:<20} | {:>8} | {:>12.0} | {:>12.1} | {:>12}",
                        format!("{}_cache_miss", name),
                        prompt_len,
                        ops_per_sec,
                        time_per_op,
                        "miss"
                    );
                    add_result("cache", result);

                    printed.store(true, Ordering::Relaxed);
                }

                duration
            });
        });

        // Benchmark cached (subsequent calls - hit)
        let cached_clone = cached_tokenizer.clone();
        cached_clone.encode(prompt).unwrap(); // Prime the cache

        let printed_hit = Arc::new(AtomicBool::new(false));
        group.bench_function(format!("{}_cache_hit", name), |b| {
            let printed = printed_hit.clone();
            let cached = cached_clone.clone();

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    black_box(cached.encode(prompt).unwrap());
                }
                let duration = start.elapsed();

                if !printed.load(Ordering::Relaxed) {
                    let ops_per_sec = iters as f64 / duration.as_secs_f64();
                    let time_per_op = duration.as_micros() as f64 / iters as f64;

                    // Get cache stats
                    let stats = cached.cache_stats().unwrap();

                    let result = format!(
                        "{:<20} | {:>8} | {:>12.0} | {:>12.1} | {:>12.1}%",
                        format!("{}_cache_hit", name),
                        prompt_len,
                        ops_per_sec,
                        time_per_op,
                        stats.hit_rate * 100.0
                    );
                    add_result("cache", result);

                    printed.store(true, Ordering::Relaxed);
                }

                duration
            });
        });
    }

    group.finish();
}

fn bench_l1_cache_chat_template(c: &mut Criterion) {
    let tokenizer_path = get_tokenizer_path();
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    // Simulate chat template with large system prompt (8KB - realistic for production)
    let system_prompt = generate_system_prompt(8000);

    // Different user queries with realistic sizes (varying from short to long)
    let user_queries = [
        SHORT_PROMPT,                  // ~30B - short question
        MEDIUM_PROMPT,                 // ~230B - detailed request
        LONG_PROMPT,                   // ~670B - complex multi-part question
        &generate_system_prompt(1000), // ~1KB - very detailed query
        &generate_system_prompt(2000), // ~2KB - extensive context
    ];

    // Create full prompts (simulating chat template)
    let prompts: Vec<String> = user_queries
        .iter()
        .map(|query| format!("{}\n\nUser: {}\nAssistant:", system_prompt, query))
        .collect();

    let mut group = c.benchmark_group("l1_cache_chat");

    // Test 0: No cache (raw tokenizer baseline)
    let printed_uncached = Arc::new(AtomicBool::new(false));
    group.bench_function("uncached_varied_prompts", |b| {
        let printed = printed_uncached.clone();
        let tokenizer = tokenizer.clone();
        let test_prompts = prompts.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                // Encode all different prompts without any cache
                for prompt in &test_prompts {
                    black_box(tokenizer.encode(prompt).unwrap());
                }
            }
            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let total_ops = iters * test_prompts.len() as u64;
                let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
                let time_per_op = duration.as_micros() as f64 / total_ops as f64;

                let result = format!(
                    "{:<25} | {:>8} | {:>12.0} | {:>12.1} | {:>20}",
                    "Uncached (baseline)",
                    test_prompts[0].len(),
                    ops_per_sec,
                    time_per_op,
                    "N/A"
                );
                add_result("l1_cache", result);

                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    // Test 1: L0-only cache (baseline)
    let l0_only_config = CacheConfig {
        enable_l0: true,
        l0_max_entries: 10_000,
        enable_l1: false,
        l1_max_memory: 0,
        l1_granularity: 128,
    };
    let cached_l0_only = Arc::new(CachedTokenizer::new(tokenizer.clone(), l0_only_config));

    let printed_l0 = Arc::new(AtomicBool::new(false));
    group.bench_function("l0_only_varied_prompts", |b| {
        let printed = printed_l0.clone();
        let cached = cached_l0_only.clone();
        let test_prompts = prompts.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                // Encode all different prompts (simulating different requests)
                for prompt in &test_prompts {
                    black_box(cached.encode(prompt).unwrap());
                }
            }
            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let total_ops = iters * test_prompts.len() as u64;
                let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
                let time_per_op = duration.as_micros() as f64 / total_ops as f64;

                let stats = cached.cache_stats().unwrap();

                let result = format!(
                    "{:<25} | {:>8} | {:>12.0} | {:>12.1} | L0:{:>6.1}% L1:{:>6}",
                    "L0-only (varied)",
                    test_prompts[0].len(),
                    ops_per_sec,
                    time_per_op,
                    stats.hit_rate * 100.0,
                    "N/A"
                );
                add_result("l1_cache", result);

                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    // Test 2: L1-only cache (prefix matching without L0)
    let l1_only_config = CacheConfig {
        enable_l0: false,
        l0_max_entries: 0,
        enable_l1: true,
        l1_max_memory: 50 * 1024 * 1024,
        l1_granularity: 128,
    };
    let cached_l1_only = Arc::new(CachedTokenizer::new(tokenizer.clone(), l1_only_config));

    let printed_l1_only = Arc::new(AtomicBool::new(false));
    group.bench_function("l1_only_varied_prompts", |b| {
        let printed = printed_l1_only.clone();
        let cached = cached_l1_only.clone();
        let test_prompts = prompts.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                // Encode all different prompts (simulating different requests)
                for prompt in &test_prompts {
                    black_box(cached.encode(prompt).unwrap());
                }
            }
            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let total_ops = iters * test_prompts.len() as u64;
                let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
                let time_per_op = duration.as_micros() as f64 / total_ops as f64;

                let l1_stats = cached.l1_cache_stats().unwrap();

                let result = format!(
                    "{:<25} | {:>8} | {:>12.0} | {:>12.1} | L0:{:>6} L1:{:>6.1}%",
                    "L1-only (varied)",
                    test_prompts[0].len(),
                    ops_per_sec,
                    time_per_op,
                    "N/A",
                    l1_stats.hit_rate * 100.0
                );
                add_result("l1_cache", result);

                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    // Test 3: L0+L1 cache (should benefit from shared prefix)
    let l0_l1_config = CacheConfig {
        enable_l0: true,
        l0_max_entries: 10_000,
        enable_l1: true,
        l1_max_memory: 50 * 1024 * 1024,
        l1_granularity: 128,
    };
    let cached_l0_l1 = Arc::new(CachedTokenizer::new(
        tokenizer.clone(),
        l0_l1_config.clone(),
    ));

    let printed_l1 = Arc::new(AtomicBool::new(false));
    group.bench_function("l0_l1_varied_prompts", |b| {
        let printed = printed_l1.clone();
        let cached = cached_l0_l1.clone();
        let test_prompts = prompts.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                // Encode all different prompts (simulating different requests)
                for prompt in &test_prompts {
                    black_box(cached.encode(prompt).unwrap());
                }
            }
            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let total_ops = iters * test_prompts.len() as u64;
                let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
                let time_per_op = duration.as_micros() as f64 / total_ops as f64;

                let stats = cached.cache_stats().unwrap();
                let l1_stats = cached.l1_cache_stats().unwrap();

                let result = format!(
                    "{:<25} | {:>8} | {:>12.0} | {:>12.1} | L0:{:>6.1}% L1:{:>6.1}%",
                    "L0+L1 (varied)",
                    test_prompts[0].len(),
                    ops_per_sec,
                    time_per_op,
                    stats.hit_rate * 100.0,
                    l1_stats.hit_rate * 100.0
                );
                add_result("l1_cache", result);

                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    // Test 4: First request cold start (L0+L1)
    let printed_cold = Arc::new(AtomicBool::new(false));
    group.bench_function("l0_l1_cold_start", |b| {
        let printed = printed_cold.clone();
        let tokenizer = tokenizer.clone();
        let first_prompt = &prompts[0];

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                // Create fresh cache for each iteration (cold start)
                let fresh_cached = CachedTokenizer::new(
                    tokenizer.clone(),
                    CacheConfig {
                        enable_l0: true,
                        l0_max_entries: 10_000,
                        enable_l1: true,
                        l1_max_memory: 50 * 1024 * 1024,
                        l1_granularity: 128,
                    },
                );
                black_box(fresh_cached.encode(first_prompt).unwrap());
            }
            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let ops_per_sec = iters as f64 / duration.as_secs_f64();
                let time_per_op = duration.as_micros() as f64 / iters as f64;

                let result = format!(
                    "{:<25} | {:>8} | {:>12.0} | {:>12.1} | {:>14}",
                    "L0+L1 (cold start)",
                    first_prompt.len(),
                    ops_per_sec,
                    time_per_op,
                    "N/A"
                );
                add_result("l1_cache", result);

                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    // Test 5: Measure prefix reuse benefit
    // First, prime the cache with the first prompt
    let cached_primed = Arc::new(CachedTokenizer::new(tokenizer.clone(), l0_l1_config));
    cached_primed.encode(&prompts[0]).unwrap(); // Prime both L0 and L1

    let printed_reuse = Arc::new(AtomicBool::new(false));
    group.bench_function("l0_l1_prefix_reuse", |b| {
        let printed = printed_reuse.clone();
        let cached = cached_primed.clone();
        let test_prompts = &prompts[1..]; // Different queries, same prefix

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                // Encode prompts with same prefix but different suffixes
                for prompt in test_prompts {
                    black_box(cached.encode(prompt).unwrap());
                }
            }
            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let total_ops = iters * test_prompts.len() as u64;
                let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
                let time_per_op = duration.as_micros() as f64 / total_ops as f64;

                let stats = cached.cache_stats().unwrap();
                let l1_stats = cached.l1_cache_stats().unwrap();

                let result = format!(
                    "{:<25} | {:>8} | {:>12.0} | {:>12.1} | L0:{:>6.1}% L1:{:>6.1}%",
                    "L0+L1 (prefix reuse)",
                    test_prompts[0].len(),
                    ops_per_sec,
                    time_per_op,
                    stats.hit_rate * 100.0,
                    l1_stats.hit_rate * 100.0
                );
                add_result("l1_cache", result);

                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    group.finish();
}

fn bench_l1_cache_production_scale(c: &mut Criterion) {
    let tokenizer_path = get_tokenizer_path();
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    let mut group = c.benchmark_group("l1_production_scale");
    group.measurement_time(Duration::from_secs(3));

    // REALISTIC Production scenario:
    // - 20k tokens total per request
    // - System prompt: 10k tokens (SHARED across many requests)
    // - User query: 10k tokens (UNIQUE per request)
    // - High prefix reuse from shared system prompts

    // Simulate 3 common system prompt sizes (matching production patterns)
    let system_prompt_10k = generate_system_prompt(40000); // ~10k tokens
    let system_prompt_5k = generate_system_prompt(20000);  // ~5k tokens
    let system_prompt_2k = generate_system_prompt(8000);   // ~2k tokens

    let system_prompts = vec![
        ("10k_system", system_prompt_10k.as_str(), 40000), // Large system prompt
        ("5k_system", system_prompt_5k.as_str(), 20000),   // Medium system prompt
        ("2k_system", system_prompt_2k.as_str(), 8000),    // Small system prompt
    ];

    println!("\n[Benchmark] Production scenario: 20k tokens/req (system prompt SHARED, user query UNIQUE)");

    for (name, system_prompt, _size) in &system_prompts {
        // Generate 100 unique user queries for this system prompt
        let mut prompts = Vec::new();
        for i in 0..100 {
            // Each user query is unique (matching total 20k tokens)
            let user_query_size = 40000; // Adjust to reach ~20k tokens total
            let user_query = format!("{} - Unique request #{}", generate_system_prompt(user_query_size), i);
            let full_prompt = format!(
                "{}\n\nUser: {}\nAssistant:",
                system_prompt, user_query
            );
            prompts.push(full_prompt);
        }

        println!("[Benchmark] {} - {} requests, avg size: ~{}KB",
                 name, prompts.len(), prompts[0].len() / 1024);

        // Test with different cache configurations for this system prompt size
        let configs = vec![
            (format!("{}_uncached", name), CacheConfig {
                enable_l0: false,
                l0_max_entries: 0,
                enable_l1: false,
                l1_max_memory: 0,
                l1_granularity: 128,
            }),
            (format!("{}_L1_50MB", name), CacheConfig {
                enable_l0: false,
                l0_max_entries: 0,
                enable_l1: true,
                l1_max_memory: 50 * 1024 * 1024,
                l1_granularity: 128,
            }),
            (format!("{}_L1_200MB", name), CacheConfig {
                enable_l0: false,
                l0_max_entries: 0,
                enable_l1: true,
                l1_max_memory: 200 * 1024 * 1024,
                l1_granularity: 128,
            }),
        ];

        for (bench_name, config) in configs {
            let is_uncached = !config.enable_l0 && !config.enable_l1;
            let tokenizer_to_use: Arc<dyn Tokenizer> = if is_uncached {
                tokenizer.clone()
            } else {
                Arc::new(CachedTokenizer::new(tokenizer.clone(), config))
            };

            let printed = Arc::new(AtomicBool::new(false));

            group.bench_function(&bench_name, |b| {
                let printed = printed.clone();
                let tok = tokenizer_to_use.clone();
                let test_prompts = prompts.clone();

                b.iter_custom(|iters| {
                    let start = Instant::now();

                    // Process all prompts to properly warm up cache and measure prefix reuse
                    for _ in 0..iters {
                        for prompt in &test_prompts {
                            black_box(tok.encode(prompt).unwrap());
                        }
                    }

                    let duration = start.elapsed();

                    if !printed.load(Ordering::Relaxed) {
                        let total_ops = iters * test_prompts.len() as u64;
                        let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
                        let time_per_op_ms = (duration.as_micros() as f64 / total_ops as f64) / 1000.0;

                        let cache_info = if is_uncached {
                            "N/A".to_string()
                        } else {
                            // Get cache stats
                            if let Some(cached) = tok.as_any().downcast_ref::<CachedTokenizer>() {
                                if let Some(l1_stats) = cached.l1_cache_stats() {
                                    format!(
                                        "Hit:{:>5.1}% Entries:{:>6} Mem:{:>6}MB",
                                        l1_stats.hit_rate * 100.0,
                                        l1_stats.entries,
                                        l1_stats.memory_bytes / (1024 * 1024)
                                    )
                                } else {
                                    "N/A".to_string()
                                }
                            } else {
                                "N/A".to_string()
                            }
                        };

                        let result = format!(
                            "{:<30} | {:>8}KB | {:>12.0} | {:>12.2}ms | {:>35}",
                            bench_name,
                            test_prompts[0].len() / 1024,
                            ops_per_sec,
                            time_per_op_ms,
                            cache_info
                        );
                        add_result("l1_production", result);

                        printed.store(true, Ordering::Relaxed);
                    }

                    duration
                });
            });
        }
    }

    group.finish();
}

fn bench_l1_cache_eviction(c: &mut Criterion) {
    let tokenizer_path = get_tokenizer_path();
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap())
            .expect("Failed to load tokenizer"),
    );

    let mut group = c.benchmark_group("l1_eviction");

    // Test with small cache that will trigger eviction
    let small_cache_config = CacheConfig {
        enable_l0: false,
        l0_max_entries: 0,
        enable_l1: true,
        l1_max_memory: 1024 * 1024, // Only 1MB - will trigger eviction
        l1_granularity: 128,
    };

    // Test with large cache that won't trigger eviction
    let large_cache_config = CacheConfig {
        enable_l0: false,
        l0_max_entries: 0,
        enable_l1: true,
        l1_max_memory: 50 * 1024 * 1024, // 50MB - no eviction
        l1_granularity: 128,
    };

    // Generate prompts with realistic production patterns:
    // - 10 different system prompts (simulating different chat contexts)
    // - 200 unique user queries
    // - Total: 2000 prompts with prefix reuse
    let num_system_prompts = 10;
    let queries_per_system = 200;

    let mut system_prompts = Vec::new();
    for i in 0..num_system_prompts {
        let sys = generate_system_prompt(4000); // 4KB each
        system_prompts.push(format!("System Context {}: {}", i, sys));
    }

    let mut prompts = Vec::new();
    for system_prompt in &system_prompts {
        for query_idx in 0..queries_per_system {
            let user_query = format!(
                "{} User request #{} with unique content details {}.",
                MEDIUM_PROMPT, query_idx, "x".repeat(100)
            );
            prompts.push(format!("{}\n\nUser: {}\nAssistant:",
                system_prompt, user_query));
        }
    }

    // Shuffle to simulate random production traffic
    use rand::seq::SliceRandom;
    let mut rng = rand::rng();
    prompts.shuffle(&mut rng);

    // Test 1: Small cache with eviction
    let cached_small = Arc::new(CachedTokenizer::new(tokenizer.clone(), small_cache_config));
    let printed_small = Arc::new(AtomicBool::new(false));

    group.bench_function("with_eviction_1mb", |b| {
        let printed = printed_small.clone();
        let cached = cached_small.clone();
        let test_prompts = prompts.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                // Encode all prompts - will trigger eviction
                for prompt in &test_prompts {
                    black_box(cached.encode(prompt).unwrap());
                }
            }
            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let total_ops = iters * test_prompts.len() as u64;
                let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
                let time_per_op = duration.as_micros() as f64 / total_ops as f64;

                let stats = cached.l1_cache_stats().unwrap();

                let result = format!(
                    "{:<25} | {:>8} | {:>12.0} | {:>12.1} | Hit:{:>6.1}% Entries:{:>6} Mem:{:>8}KB",
                    "L1 (1MB, eviction)",
                    test_prompts[0].len(),
                    ops_per_sec,
                    time_per_op,
                    stats.hit_rate * 100.0,
                    stats.entries,
                    stats.memory_bytes / 1024
                );
                add_result("l1_eviction", result);

                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    // Test 2: Large cache without eviction
    let cached_large = Arc::new(CachedTokenizer::new(tokenizer.clone(), large_cache_config));
    let printed_large = Arc::new(AtomicBool::new(false));

    group.bench_function("no_eviction_50mb", |b| {
        let printed = printed_large.clone();
        let cached = cached_large.clone();
        let test_prompts = prompts.clone();

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                // Encode all prompts - no eviction needed
                for prompt in &test_prompts {
                    black_box(cached.encode(prompt).unwrap());
                }
            }
            let duration = start.elapsed();

            if !printed.load(Ordering::Relaxed) {
                let total_ops = iters * test_prompts.len() as u64;
                let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
                let time_per_op = duration.as_micros() as f64 / total_ops as f64;

                let stats = cached.l1_cache_stats().unwrap();

                let result = format!(
                    "{:<25} | {:>8} | {:>12.0} | {:>12.1} | Hit:{:>6.1}% Entries:{:>6} Mem:{:>8}KB",
                    "L1 (50MB, no eviction)",
                    test_prompts[0].len(),
                    ops_per_sec,
                    time_per_op,
                    stats.hit_rate * 100.0,
                    stats.entries,
                    stats.memory_bytes / 1024
                );
                add_result("l1_eviction", result);

                printed.store(true, Ordering::Relaxed);
            }

            duration
        });
    });

    group.finish();
}

// Print final summary table
fn print_summary() {
    println!("\n{}", "=".repeat(120));
    println!("TOKENIZER BENCHMARK SUMMARY");
    println!("{}", "=".repeat(120));

    let results = BENCHMARK_RESULTS.lock().unwrap();

    let mut current_category = String::new();
    for (key, value) in results.iter() {
        let category = key.split('_').skip(1).collect::<Vec<_>>().join("_");

        if category != current_category {
            current_category = category.clone();

            // Print section header based on category
            println!("\n{}", "-".repeat(120));
            match category.as_str() {
                "encode" => {
                    println!("ENCODING THROUGHPUT");
                    println!(
                        "{:<15} | {:>8} | {:>8} | {:>12} | {:>12} | {:>10} | {:>10}",
                        "Test Case",
                        "Size(B)",
                        "Tokens",
                        "Chars/sec",
                        "Tokens/sec",
                        "Ops/sec",
                        "Thread"
                    );
                }
                "batch" => {
                    println!("BATCH ENCODING");
                    println!(
                        "{:<15} | {:>8} | {:>8} | {:>12} | {:>12} | {:>10} | {:>10}",
                        "Batch Size",
                        "Size(B)",
                        "Tokens",
                        "Prompts/sec",
                        "Tokens/sec",
                        "Chars/sec",
                        "Thread"
                    );
                }
                "concurrent" => {
                    println!("CONCURRENT ENCODING");
                    println!(
                        "{:<15} | {:>10} | {:>12} | {:>12} | {:>15}",
                        "Clients", "Total Ops", "Ops/sec", "Chars/sec", "Per-Client/sec"
                    );
                }
                "mt_encode" => {
                    println!("MULTI-THREADED ENCODING");
                    println!(
                        "{:<20} | {:>10} | {:>12} | {:>12} | {:>10} | {:>12}",
                        "Configuration",
                        "Total Ops",
                        "Ops/sec",
                        "Tokens/sec",
                        "Threads",
                        "Efficiency"
                    );
                }
                "decode" => {
                    println!("DECODE PERFORMANCE");
                    println!(
                        "{:<20} | {:>10} | {:>12} | {:>12} | {:>10}",
                        "Method", "Tokens", "Tokens/sec", "Ops/sec", "Thread"
                    );
                }
                "mt_decode" => {
                    println!("MULTI-THREADED DECODING");
                    println!(
                        "{:<20} | {:>12} | {:>12} | {:>10} | {:>12}",
                        "Configuration", "Total Tokens", "Tokens/sec", "Threads", "Efficiency"
                    );
                }
                "streaming_100k" => {
                    println!("STREAMING DECODE (100K Target)");
                    println!(
                        "{:<20} | {:>12} | {:>12} | {:>12} | {:>10} | {:>12}",
                        "Method", "Tokens", "Tokens/sec", "Target", "Thread", "Status"
                    );
                }
                "concurrent_streaming" => {
                    println!("CONCURRENT STREAMING");
                    println!(
                        "{:<20} | {:>10} | {:>12} | {:>15} | {:>15}",
                        "Sequences", "Total", "Aggregate/sec", "Per-Seq/sec", "Threads"
                    );
                }
                "stop_sequences" => {
                    println!("STOP SEQUENCE PERFORMANCE");
                    println!(
                        "{:<20} | {:>10} | {:>12} | {:>12} | {:>10}",
                        "Config", "Sequences", "Tokens", "Tokens/sec", "Seq/sec"
                    );
                }
                "latency" => {
                    println!("LATENCY DISTRIBUTION");
                    println!(
                        "{:<20} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}",
                        "Operation", "P50(µs)", "P95(µs)", "P99(µs)", "Max(µs)", "Samples"
                    );
                }
                "scaling" => {
                    println!("SCALING CHARACTERISTICS");
                    println!(
                        "{:<15} | {:>12} | {:>12} | {:>12}",
                        "Threads", "Total Tokens", "Tokens/sec", "Efficiency"
                    );
                }
                "memory" => {
                    println!("MEMORY EFFICIENCY");
                    println!(
                        "{:<20} | {:>12} | {:>12} | {:>12}",
                        "Operation", "Calls/sec", "Time/call", "Improvement"
                    );
                }
                "cache" => {
                    println!("CACHE PERFORMANCE (L0 WHOLE-STRING)");
                    println!(
                        "{:<20} | {:>8} | {:>12} | {:>12} | {:>12}",
                        "Test Case", "Size(B)", "Ops/sec", "Time(µs)", "Status"
                    );
                }
                "l1_cache" => {
                    println!("L1 CACHE (PREFIX MATCHING) - CHAT TEMPLATE");
                    println!(
                        "{:<25} | {:>8} | {:>12} | {:>12} | {:>20}",
                        "Test Case", "Size(B)", "Ops/sec", "Time(µs)", "Hit Rates"
                    );
                }
                "l1_eviction" => {
                    println!("L1 CACHE EVICTION (PRODUCTION STRESS TEST)");
                    println!(
                        "{:<25} | {:>8} | {:>12} | {:>12} | {:>50}",
                        "Test Case", "Size(B)", "Ops/sec", "Time(µs)", "Cache Stats"
                    );
                }
                "l1_production" => {
                    println!("L1 CACHE PRODUCTION SCALE (20K TOKENS: SHARED SYSTEM + UNIQUE USER)");
                    println!(
                        "{:<30} | {:>9} | {:>12} | {:>13} | {:>35}",
                        "Configuration", "Size", "Req/sec", "Time/req", "Cache Stats"
                    );
                }
                _ => {}
            }
            println!("{}", "-".repeat(120));
        }

        println!("{}", value);
    }

    println!("\n{}", "=".repeat(120));
}

fn run_benchmarks(c: &mut Criterion) {
    bench_encode_throughput(c);
    bench_batch_encode(c);
    bench_concurrent_encode(c);
    bench_multithreaded_encode(c);
    bench_decode_performance(c);
    bench_multithreaded_decode(c);
    bench_streaming_decode_100k(c);
    bench_concurrent_streaming(c);
    bench_stop_sequences(c);
    bench_latency_distribution(c);
    bench_scaling_characteristics(c);
    bench_memory_efficiency(c);
    bench_cached_vs_uncached(c);
    bench_l1_cache_chat_template(c);
    bench_l1_cache_eviction(c);
    bench_l1_cache_production_scale(c);

    // Print summary at the end
    print_summary();
}

criterion_group!(benches, run_benchmarks);
criterion::criterion_main!(benches);
