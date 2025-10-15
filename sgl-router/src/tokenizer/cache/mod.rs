//! Tokenizer Caching Layer
//!
//! Provides a caching wrapper around any tokenizer implementation to speed up
//! repeated tokenization of the same strings (e.g., system prompts).
//!
//! # Architecture
//! - **L0 Cache**: Whole-string exact match (90% of wins)
//! - **L1 Cache**: Prefix matching at fixed boundaries (future work)
//!
//! # Usage
//! ```ignore
//! let tokenizer = Arc::new(HuggingFaceTokenizer::from_file("tokenizer.json")?);
//! let cached = Arc::new(CachedTokenizer::new(tokenizer, CacheConfig::default()));
//! let encoding = cached.encode("Hello world")?;
//! ```

mod fingerprint;
mod l0;

pub use fingerprint::TokenizerFingerprint;
pub use l0::{CacheStats, L0Cache};

use super::traits::{Decoder, Encoder, Encoding, SpecialTokens, TokenIdType, Tokenizer};
use anyhow::Result;
use std::sync::Arc;

/// Configuration for the tokenizer cache
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Enable L0 (whole-string) cache
    pub enable_l0: bool,
    /// Maximum number of entries in L0 cache
    pub l0_max_entries: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enable_l0: true,
            l0_max_entries: 10_000, // ~22MB memory for typical prompts
        }
    }
}

/// A caching wrapper around any tokenizer
pub struct CachedTokenizer {
    /// The underlying tokenizer
    inner: Arc<dyn Tokenizer>,
    /// L0 cache (whole-string exact match)
    l0: Option<L0Cache>,
    /// Configuration
    #[allow(dead_code)]
    config: CacheConfig,
    /// Fingerprint for cache invalidation
    fingerprint: TokenizerFingerprint,
}

impl CachedTokenizer {
    /// Create a new cached tokenizer
    pub fn new(inner: Arc<dyn Tokenizer>, config: CacheConfig) -> Self {
        let fingerprint = TokenizerFingerprint::from_tokenizer(inner.as_ref());

        let l0 = if config.enable_l0 {
            Some(L0Cache::new(config.l0_max_entries))
        } else {
            None
        };

        Self {
            inner,
            l0,
            config,
            fingerprint,
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> Option<CacheStats> {
        self.l0.as_ref().map(|cache| cache.stats())
    }

    /// Clear the cache
    pub fn clear_cache(&self) {
        if let Some(l0) = &self.l0 {
            l0.clear();
        }
    }

    /// Get the fingerprint of the underlying tokenizer
    pub fn fingerprint(&self) -> &TokenizerFingerprint {
        &self.fingerprint
    }
}

impl Encoder for CachedTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        // L0 cache lookup
        if let Some(l0) = &self.l0 {
            if let Some(cached) = l0.get(input) {
                return Ok(cached);
            }
        }

        // Cache miss - tokenize and cache result
        let encoding = self.inner.encode(input)?;

        // Cache the result
        if let Some(l0) = &self.l0 {
            l0.insert(input.to_string(), encoding.clone());
        }

        Ok(encoding)
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        // Process each input independently
        inputs.iter().map(|&input| self.encode(input)).collect()
    }
}

impl Decoder for CachedTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String> {
        // Decoding is not cached (it's fast enough and rarely repeated)
        self.inner.decode(token_ids, skip_special_tokens)
    }
}

impl Tokenizer for CachedTokenizer {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn get_special_tokens(&self) -> &SpecialTokens {
        self.inner.get_special_tokens()
    }

    fn token_to_id(&self, token: &str) -> Option<TokenIdType> {
        self.inner.token_to_id(token)
    }

    fn id_to_token(&self, id: TokenIdType) -> Option<String> {
        self.inner.id_to_token(id)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::mock::MockTokenizer;

    #[test]
    fn test_cache_hit() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let cached = CachedTokenizer::new(tokenizer, CacheConfig::default());

        let input = "Hello world";

        // First call - miss
        let result1 = cached.encode(input).unwrap();

        // Second call - hit
        let result2 = cached.encode(input).unwrap();

        // Results should be identical
        assert_eq!(result1.token_ids(), result2.token_ids());

        // Check cache stats
        let stats = cached.cache_stats().unwrap();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_cache_disabled() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let config = CacheConfig {
            enable_l0: false,
            l0_max_entries: 0,
        };
        let cached = CachedTokenizer::new(tokenizer, config);

        let input = "Hello world";

        // Both calls should work even without cache
        let result1 = cached.encode(input).unwrap();
        let result2 = cached.encode(input).unwrap();

        assert_eq!(result1.token_ids(), result2.token_ids());

        // No cache stats available
        assert!(cached.cache_stats().is_none());
    }

    #[test]
    fn test_encode_batch() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let cached = CachedTokenizer::new(tokenizer, CacheConfig::default());

        let inputs = vec!["Hello", "world", "Hello"]; // "Hello" repeated

        let results = cached.encode_batch(&inputs).unwrap();

        assert_eq!(results.len(), 3);

        // Check cache stats - should have 1 hit for the repeated "Hello"
        let stats = cached.cache_stats().unwrap();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 2);
    }

    #[test]
    fn test_decoder_passthrough() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let cached = CachedTokenizer::new(tokenizer, CacheConfig::default());

        let tokens = vec![1, 2, 3];
        let decoded = cached.decode(&tokens, false).unwrap();

        // Should just pass through to inner tokenizer
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_tokenizer_trait_methods() {
        let tokenizer = Arc::new(MockTokenizer::new());
        let cached = CachedTokenizer::new(tokenizer.clone(), CacheConfig::default());

        // Should pass through to inner tokenizer
        assert_eq!(cached.vocab_size(), tokenizer.vocab_size());
        assert!(cached.token_to_id("Hello").is_some());
        assert!(cached.id_to_token(1).is_some());
    }
}
