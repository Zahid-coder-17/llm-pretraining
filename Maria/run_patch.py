import glob, os
from collections import Counter
files = glob.glob('/workspace/final_115B/*.parquet')
prefixes = Counter([os.path.basename(f).split('_')[0] for f in files])
for p, c in prefixes.items():
    sample = [os.path.basename(f) for f in files if os.path.basename(f).startswith(p)][0]
    print(f"Prefix '{p}': {c} files | Sample: {sample}")
