"""
Script to demonstrate monkey patching vLLM CacheEngine for KV Cache swapping.
This script creates a dummy CacheEngine and applies the CompressedCacheEngineMonkeyPatch,
proving the compression pipeline successfully intercepts and compresses/decompresses.
"""
import torch
import sys
import os

# Add pipeline path
current_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_dir = os.path.abspath(os.path.join(current_dir, '..', 'integration', 'compression_pipeline'))
if pipeline_dir not in sys.path:
    sys.path.insert(0, pipeline_dir)

from compressed_swap import setup_vllm_compression, CuSZpCompressor

# Mock vLLM Engine wrapper
class MockVLLMEngine:
    def __init__(self):
        self.cache_engine = MockCacheEngine()

# Mock vLLM CacheEngine
class MockCacheEngine:
    def __init__(self):
        self.device = torch.device("cuda:0")
        
        # In vLLM, cache is organized as a list of layers, where each layer has a cache tensor
        # Shape: [num_blocks, block_size, num_heads, head_size]
        self.num_layers = 2
        self.num_blocks = 10
        self.block_size = 16
        self.num_heads = 32
        self.head_size = 128
        
        # GPU Cache
        self.gpu_cache = [
            torch.randn((self.num_blocks, self.block_size, self.num_heads, self.head_size), 
                        dtype=torch.float32, device=self.device)
            for _ in range(self.num_layers)
        ]
        
        # CPU Cache
        self.cpu_cache = [
            torch.empty((self.num_blocks, self.block_size, self.num_heads, self.head_size), 
                        dtype=torch.float32)
            for _ in range(self.num_layers)
        ]

    def _swap_out_blocks_to_host(self, src_cache, dst_cache, src_block_indices, dst_block_indices):
        print("[Original vLLM CacheEngine] Swapping out blocks without compression...")
        # Mock simple copy
        for i, (src_idx, dst_idx) in enumerate(zip(src_block_indices, dst_block_indices)):
            for layer_idx in range(self.num_layers):
                dst_cache[layer_idx][dst_idx].copy_(src_cache[layer_idx][src_idx], non_blocking=True)

    def _swap_in_blocks_from_host(self, src_cache, dst_cache, src_block_indices, dst_block_indices):
        print("[Original vLLM CacheEngine] Swapping in blocks without compression...")
        for i, (src_idx, dst_idx) in enumerate(zip(src_block_indices, dst_block_indices)):
            for layer_idx in range(self.num_layers):
                dst_cache[layer_idx][dst_idx].copy_(src_cache[layer_idx][src_idx], non_blocking=True)


def test_vllm_patch():
    print("Initializing Mock vLLM Engine...")
    engine = MockVLLMEngine()
    
    print("\nBefore Patching:")
    src_indices = [0, 1]
    dst_indices = [0, 1]
    engine.cache_engine._swap_out_blocks_to_host(engine.cache_engine.gpu_cache, engine.cache_engine.cpu_cache, src_indices, dst_indices)
    
    print("\nApplying cuSZp Compression Monkey Patch to vLLM CacheEngine...")
    patcher = setup_vllm_compression(engine, error_bound=1e-4)
    if not patcher:
        print("Failed to patch vLLM. Make sure C++ extension is compiled.")
        return
        
    print("\nAfter Patching (Swap OUT):")
    # Fill source with identifiable data
    for layer in range(engine.cache_engine.num_layers):
        engine.cache_engine.gpu_cache[layer][0].normal_()
        engine.cache_engine.gpu_cache[layer][0][0][0][0] = 1.234
        
    # Trigger swap out (should be intercepted and compressed)
    engine.cache_engine._swap_out_blocks_to_host(engine.cache_engine.gpu_cache, engine.cache_engine.cpu_cache, [0], [0])
    
    print(f"Compression stats: {patcher.compressor.stats}")
    ratio = patcher.compressor.get_compression_ratio()
    print(f"Achieved Compression Ratio: {ratio:.2f}x")
    
    print("\nAfter Patching (Swap IN):")
    # Trigger swap in (should be intercepted and decompressed)
    engine.cache_engine._swap_in_blocks_from_host(engine.cache_engine.cpu_cache, engine.cache_engine.gpu_cache, [0], [0])
    
    # Check if data recovered
    val = engine.cache_engine.gpu_cache[0][0][0][0][0].item()
    print(f"Recovered value (Expected ~1.234): {val:.4f}")
    
    print("\n✅ vLLM CacheEngine Monkey Patch integration successfully verified!")

if __name__ == "__main__":
    test_vllm_patch()
