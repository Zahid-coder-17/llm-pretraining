import os, glob
import pyarrow.parquet as pq

domains = {
    "general": ("general_", 45.0),
    "code": ("code_", 20.0),
    "edu": ("edu_", 25.0),
    "multilingual": ("multilingual_", 15.0),
    "science_math": ("math_", 10.0),
}

base_dir = "/workspace/final_115B"
all_files = glob.glob(os.path.join(base_dir, "*.parquet"))
print(f"Total Parquet files found on disk: {len(all_files):,}")

stats = {k: {"files": 0, "bytes": 0, "tokens": 0, "target": v[1]} for k, v in domains.items()}
other_files = 0

for f in all_files:
    fname = os.path.basename(f)
    sz = os.path.getsize(f)
    matched = False
    for k, v in domains.items():
        if fname.startswith(v[0]):
            stats[k]["files"] += 1
            stats[k]["bytes"] += sz
            matched = True
            break
    if not matched:
        other_files += 1

print(f"{'DOMAIN':<15} | {'FILES':<8} | {'SIZE (GB)':<10} | {'TARGET (B)':<10}")
print("-" * 52)
for k, s in stats.items():
    gb = s["bytes"] / (1024**3)
    print(f"{k:<15} | {s['files']:<8} | {gb:<10.2f} | {s['target']:<10.1f}")
print("-" * 52)
total_gb = sum(s["bytes"] for s in stats.values()) / (1024**3)
print(f"{'TOTAL':<15} | {len(all_files):<8} | {total_gb:<10.2f} | 115.0")
