import glob, os
import pyarrow.parquet as pq

domains = {
    "general": ("general_", 2969),
    "code": ("code_", 1875),
    "edu": ("edu_", 2161),
    "multilingual": ("multilingual_", 993),
    "science_math": ("science_math_", 360),
}

print(f"{'DOMAIN':<15} | {'FILES':<6} | {'ROWS/FILE':<10} | {'APPROX TOKENS':<15}")
print("-" * 55)

for dom, (prefix, fcount) in domains.items():
    sample_files = glob.glob(f"/workspace/final_115B/{prefix}*.parquet")
    if sample_files:
        md = pq.read_metadata(sample_files[0])
        rows_per_file = md.num_rows
        approx_tok_b = (rows_per_file * fcount) / 1e9
        print(f"{dom:<15} | {fcount:<6} | {rows_per_file:<10,d} | {approx_tok_b:<14.3f} B")
