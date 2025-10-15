//! L1 Cache: Fixed-boundary prefix cache
//!
//! Caches tokenization results at fixed byte boundaries (e.g., every 128 bytes).
//! Useful for chat templates where different requests share common prefixes.
//!
//! Example:
//! ```
//! Template: "<|system|>You are helpful.<|end|><|user|>{query}<|end|>"
//!
//! Request 1: "<|system|>You are helpful.<|end|><|user|>What is 2+2?<|end|>"
//! Request 2: "<|system|>You are helpful.<|end|><|user|>Hello!<|end|>"
//!
//! The prefix "<|system|>You are helpful.<|end|><|user|>" can be cached
//! at the 128-byte boundary if it's longer than 128 bytes.
//! ```

use super::super::traits::TokenIdType;
use blake3;
use dashmap::DashMap;
use std::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Hash type for cache keys
type Blake3Hash = [u8; 32];

/// Number of shards for concurrent access
const NUM_SHARDS: usize = 16;

/// A cached prefix entry
#[derive(Debug, Clone)]
struct CachedPrefix {
    /// The token IDs for this prefix
    tokens: Vec<TokenIdType>,
}

/// L1 cache implementation with fixed-boundary prefix matching
pub struct L1Cache {
    /// Sharded maps for concurrent access
    /// Key: Blake3 hash of bytes[0..boundary]
    /// Value: Cached token IDs for that prefix
    shards: Vec<Arc<DashMap<Blake3Hash, CachedPrefix>>>,
    /// Granularity in bytes (e.g., 128)
    granularity: usize,
    /// Maximum memory in bytes
    max_memory: usize,
    /// Current memory usage estimate
    current_memory: AtomicU64,
    /// Cache hit counter
    hits: AtomicU64,
    /// Cache miss counter
    misses: AtomicU64,
}

impl L1Cache {
    /// Create a new L1 cache
    pub fn new(max_memory: usize, granularity: usize) -> Self {
        let shards = (0..NUM_SHARDS).map(|_| Arc::new(DashMap::new())).collect();

        Self {
            shards,
            granularity,
            max_memory,
            current_memory: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Try to find the longest prefix match for the given input
    /// Returns (cached_tokens, byte_offset) if found
    pub fn longest_prefix_match(&self, input: &str) -> Option<(Vec<TokenIdType>, usize)> {
        let bytes = input.as_bytes();

        // Calculate the maximum boundary we can check
        let max_boundary = (bytes.len() / self.granularity) * self.granularity;

        if max_boundary == 0 {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        // Search backwards from the largest boundary to find longest match
        // Manually iterate in reverse to avoid .rev() issue with RangeInclusive<usize>
        let num_boundaries = max_boundary / self.granularity;
        for i in (1..=num_boundaries).rev() {
            let k = i * self.granularity;
            let prefix = &bytes[0..k];
            let hash = blake3::hash(prefix);
            let hash_bytes: Blake3Hash = *hash.as_bytes();

            let shard_idx = hash_bytes[0] as usize % NUM_SHARDS;

            if let Some(entry) = self.shards[shard_idx].get(&hash_bytes) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Some((entry.tokens.clone(), k));
            }
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Insert prefix entries at all boundaries for the given input
    pub fn insert_at_boundaries(&self, input: &str, tokens: &[TokenIdType]) {
        let bytes = input.as_bytes();

        // Don't cache if we're over memory limit
        if self.current_memory.load(Ordering::Relaxed) as usize >= self.max_memory {
            return;
        }

        // Insert at each boundary
        for k in (self.granularity..bytes.len()).step_by(self.granularity) {
            let prefix = &bytes[0..k];
            let hash = blake3::hash(prefix);
            let hash_bytes: Blake3Hash = *hash.as_bytes();

            let shard_idx = hash_bytes[0] as usize % NUM_SHARDS;

            // For this prefix, we need to know how many tokens it represents
            // We'll use a simple heuristic: assume uniform token distribution
            let token_ratio = tokens.len() as f64 / bytes.len() as f64;
            let estimated_tokens = (k as f64 * token_ratio) as usize;
            let prefix_tokens = tokens[..estimated_tokens.min(tokens.len())].to_vec();

            let size_bytes = k + prefix_tokens.len() * size_of::<TokenIdType>();
            let cached = CachedPrefix {
                tokens: prefix_tokens,
            };

            self.shards[shard_idx].insert(hash_bytes, cached);
            self.current_memory
                .fetch_add(size_bytes as u64, Ordering::Relaxed);
        }
    }

    /// Get the number of entries in the cache
    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.shards.iter().all(|s| s.is_empty())
    }

    /// Get cache statistics
    pub fn stats(&self) -> L1CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total_requests = hits + misses;

        L1CacheStats {
            hits,
            misses,
            entries: self.len(),
            memory_bytes: self.current_memory.load(Ordering::Relaxed) as usize,
            hit_rate: if total_requests > 0 {
                hits as f64 / total_requests as f64
            } else {
                0.0
            },
        }
    }

    /// Clear the cache
    pub fn clear(&self) {
        for shard in &self.shards {
            shard.clear();
        }
        self.current_memory.store(0, Ordering::Relaxed);
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone)]
pub struct L1CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub entries: usize,
    pub memory_bytes: usize,
    pub hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_prefix_match() {
        let cache = L1Cache::new(1024 * 1024, 128);

        let input1 = "a".repeat(200); // 200 bytes
        let tokens1 = vec![1, 2, 3, 4, 5];

        // Insert at boundaries
        cache.insert_at_boundaries(&input1, &tokens1);

        // Should have cached at 128-byte boundary
        assert!(!cache.is_empty());

        // Search with same prefix
        let input2 = format!("{}{}", &input1[..128], "different suffix");
        let result = cache.longest_prefix_match(&input2);

        assert!(result.is_some());
        let (tokens, offset) = result.unwrap();
        assert_eq!(offset, 128);
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_no_match_below_granularity() {
        let cache = L1Cache::new(1024 * 1024, 128);

        let input = "short"; // Only 5 bytes, below granularity
        let tokens = vec![1, 2];

        cache.insert_at_boundaries(input, &tokens);

        // Should not cache anything (too short)
        assert!(cache.is_empty());

        // Should return None
        let result = cache.longest_prefix_match(input);
        assert!(result.is_none());
    }

    #[test]
    fn test_longest_match() {
        let cache = L1Cache::new(1024 * 1024, 128);

        let input = "a".repeat(400); // 400 bytes -> boundaries at 128, 256, 384
        let tokens = vec![1; 100];

        cache.insert_at_boundaries(&input, &tokens);

        // Should have 3 entries (128, 256, 384)
        assert_eq!(cache.len(), 3);

        // Search with 300 bytes - should match 256 boundary
        let search_input = "a".repeat(300);
        let result = cache.longest_prefix_match(&search_input);

        assert!(result.is_some());
        let (_, offset) = result.unwrap();
        assert_eq!(offset, 256); // Longest match
    }

    #[test]
    fn test_stats() {
        let cache = L1Cache::new(1024 * 1024, 128);

        let input = "a".repeat(200);
        let tokens = vec![1, 2, 3];

        cache.insert_at_boundaries(&input, &tokens);

        // Try to find match
        let _ = cache.longest_prefix_match(&input);

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.hit_rate, 1.0);
    }

    #[test]
    fn test_clear() {
        let cache = L1Cache::new(1024 * 1024, 128);

        let input = "a".repeat(200);
        let tokens = vec![1, 2, 3];

        cache.insert_at_boundaries(&input, &tokens);
        assert!(!cache.is_empty());

        cache.clear();
        assert!(cache.is_empty());

        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }
}
