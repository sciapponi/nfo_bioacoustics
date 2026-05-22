"""
Attempt to load WhaleSounds dataset using different methods.
"""

from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

print("Attempting to load WhaleSounds with various methods...\n")

# Method 1: Try with streaming
print("Method 1: Streaming mode...")
try:
    ds = load_dataset("monster-monash/WhaleSounds", streaming=True)
    print(f"✓ Streaming works! Dataset structure: {ds}")
    print(f"  Keys: {list(ds.keys())}")
except Exception as e:
    print(f"✗ Streaming failed: {e}\n")

# Method 2: Check dataset info
print("\nMethod 2: Get dataset info...")
try:
    from huggingface_hub import dataset_info
    info = dataset_info("monster-monash/WhaleSounds")
    print(f"✓ Dataset info retrieved")
    print(f"  Siblings: {info.siblings}")
except Exception as e:
    print(f"✗ Info retrieval failed: {e}\n")

# Method 3: List dataset files
print("\nMethod 3: List available files...")
try:
    from huggingface_hub import list_repo_files
    files = list_repo_files(repo_id="monster-monash/WhaleSounds", repo_type="dataset")
    print(f"✓ Files found: {len(files)}")
    for f in sorted(files)[:20]:  # Show first 20
        print(f"  - {f}")
except Exception as e:
    print(f"✗ File listing failed: {e}\n")

# Method 4: Try alternative config names
print("\nMethod 4: Try alternative configurations...")
try:
    configs = load_dataset("monster-monash/WhaleSounds", name="default")
    print(f"✓ Config 'default' works!")
except Exception as e:
    print(f"✗ Config attempt failed: {e}")

print("\n" + "="*60)
print("If Method 3 shows parquet files, the dataset might be loadable")
print("If Method 1 (streaming) works, we can use that approach")
