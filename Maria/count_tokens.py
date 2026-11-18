import glob, os
import pyarrow.parquet as pq

domain_map = {
    "general": "general_",
    "edu": "edu_",
    "code": "code_",
    "multilingual": "multilingual_",
    "science_math": "science_math_",
}

files = glob.glob('/workspace/final_115B/*.parquet')
stats = {k: {"files": 0, "rows": 0, "bytes": 0} for k in domain_map}

for f in files:
    fname = os.path.basename(f)
    sz = os.path.getsize(f)
    for dom, prefix in domain_map.items():
        if fname.startswith(prefix):
            stats[dom]["files"] += 1
            stats[dom]["bytes"] += sz
            try:
                md = pq.read_metadata(f)
                stats[dom]["rows"] += md.num_rows
            except Exception:
                pass
            break

print("=" * 70)
print(f"{'DOMAIN':<15} | {'FILES':<6} | {'SIZE (GB)':<9} | {'AVAILABLE TOKENS':<16} | {'PERCENT':<8}")
print("-" * 70)
total_tokens = sum(s["rows"] for s in stats.values())
total_gb = sum(s["bytes"] for s in stats.values()) / (1024**3)

for dom, s in stats.items():
    tok_b = s["rows"] / 1e9
    gb = s["bytes"] / (1024**3)
    pct = (s["rows"] / total_tokens * 100) if total_tokens > 0 else 0
    print(f"{dom:<15} | {s['files']:<6} | {gb:<9.2f} | {tok_b:<14.3f} B | {pct:<7.2f}%")

print("-" * 70)
print(f"{'TOTAL':<15} | {len(files):<6} | {total_gb:<9.2f} | {total_tokens/1e9:<14.3f} B | 100.00%")
print("=" * 70)
