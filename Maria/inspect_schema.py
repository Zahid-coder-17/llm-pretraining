import glob
import pyarrow.parquet as pq

f = glob.glob("/workspace/final_115B/general_*.parquet")[0]
table = pq.read_table(f)
print("Table Schema:", table.schema)
print("Num Rows:", table.num_rows)
col = table.column(0)
sample_elem = col[0].as_py()
print("First Element Type:", type(sample_elem))
if isinstance(sample_elem, list):
    print("Element length:", len(sample_elem))
else:
    print("First 10 values:", col[:10].to_pylist())
