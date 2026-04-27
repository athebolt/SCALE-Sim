import pandas as pd, io

with open('ncu-metrics/ncu_hardware_metrics.csv') as f:
    lines = f.readlines()

header_idx = None
for i, line in enumerate(lines):
    if line.startswith('"ID"'):
        header_idx = i
        break

csv_text = ''.join(lines[header_idx:])
df = pd.read_csv(io.StringIO(csv_text))
df.columns = [c.strip() for c in df.columns]

def sf(v):
    if pd.isna(v): return float('nan')
    s = str(v).replace(',','').strip()
    if s in ('n/a','N/A',''): return float('nan')
    try: return float(s)
    except: return float('nan')

df['value'] = df['Metric Value'].apply(sf)
pivot = df.pivot_table(index=['ID','Kernel Name'], columns='Metric Name', values='value', aggfunc='first').reset_index()

conv_kw = ['cudnn','gemm','conv','fprop','xmma','cutlass','sgemm']
pivot['is_conv'] = pivot['Kernel Name'].apply(lambda n: any(k in n.lower() for k in conv_kw))
conv = pivot[pivot['is_conv']]

cycles_col = [c for c in pivot.columns if 'sm__cycles_active.avg' in str(c)][0]
l1_col = [c for c in pivot.columns if 'l1tex__t_sectors.sum' in str(c)][0]
l2_col = [c for c in pivot.columns if 'lts__t_sectors.sum' in str(c)][0]

# Find the pattern: how many times does each unique kernel name appear?
name_counts = conv['Kernel Name'].value_counts()
print("Kernel name repetition counts:")
for name, cnt in name_counts.items():
    print(f"  x{cnt:3d}  {name[:90]}")

# Try to determine the number of inference passes
# The total number of conv kernels should be N_layers * num_passes
# ResNet18 has 21 layers, 168 conv kernels -> 8 passes
total_conv = len(conv)
print(f"\nTotal conv kernels: {total_conv}")
print(f"Implied passes (÷21): {total_conv/21:.1f}")

# Compute per-pass averages
# Group by kernel name and average the cycles
avg_by_name = conv.groupby('Kernel Name')[cycles_col].agg(['mean','sum','count'])
print(f"\nPer-kernel-name averages:")
total_avg = 0
for name, row in avg_by_name.iterrows():
    print(f"  avg={row['mean']:>10,.0f}  sum={row['sum']:>12,.0f}  cnt={row['count']:2.0f}  {name[:80]}")
    total_avg += row['mean']

print(f"\nSum of per-name averages: {total_avg:,.0f}")
print(f"This would be the single-pass cycle estimate")

# Group unique conv kernels (deduplicated) and compute single-pass totals
# Assume each unique kernel maps to one layer
print(f"\n--- Single-pass estimate (average of each unique kernel) ---")
single_pass_cycles = avg_by_name['mean'].sum()
single_pass_l1 = conv.groupby('Kernel Name')[l1_col].mean().sum()
single_pass_l2 = conv.groupby('Kernel Name')[l2_col].mean().sum()
print(f"  Cycles: {single_pass_cycles:,.0f}")
print(f"  L1 sectors: {single_pass_l1:,.0f}")
print(f"  L2 sectors: {single_pass_l2:,.0f}")
print(f"  Total on-chip sectors: {single_pass_l1 + single_pass_l2:,.0f}")
