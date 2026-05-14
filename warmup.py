#!/usr/bin/env python
"""Warmup script to pre-load the LLM model at startup"""
import sys
import time

print("Warming up local LLM model...")
start = time.time()

from rag.local_llm import get_local_llm

try:
    llm = get_local_llm()
    if llm.pipeline is None:
        print("ERROR: Model failed to load!")
        sys.exit(1)
    
    # Test a simple inference
    test_prompt = "Answer: The quick brown fox"
    result = llm.generate_response(test_prompt)
    
    elapsed = time.time() - start
    print(f"✅ Model warmup complete in {elapsed:.2f}s")
    print(f"✅ Test inference: {result[:50]}...")
    
except Exception as e:
    print(f"ERROR during warmup: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
