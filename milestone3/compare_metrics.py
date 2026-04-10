#!/usr/bin/env python3
"""
compare_metrics.py

Compares hardware profiling metrics from NVIDIA Nsight Compute (NCU)
with cycle-accurate simulation results from SCALE-Sim v2.

Outputs:
  - A per-kernel summary of hardware metrics
  - A per-layer summary of SCALE-Sim metrics
  - An aggregate comparison table
  - A combined report saved to comparison_report.txt
"""

import pandas as pd
import os
import sys
import io


# ---------------------------------------------------------------------------
# 1. Parse NCU CSV
# ---------------------------------------------------------------------------
def parse_ncu_csv(path):
    """Parse the real NCU --csv output.

    Each row is one (kernel-invocation, metric) pair. 
    We aggregate across ALL kernel invocations to obtain totals.
    """

    # ensure the file exists
    if not os.path.exists(path):
        print(f"  [!] NCU file not found: {path}")
        return None

    # Find the CSV header row
    with open(path, "r") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('"ID"'):
            header_idx = i
            break

    if header_idx is None:
        print("  [!] Could not locate CSV header in NCU output.")
        return None

    csv_text = "".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(csv_text))

    # Normalise column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Build per-kernel aggregates -----------------------------------------------
    # Each kernel invocation appears in multiple rows (one per metric).
    # Group by (ID, Kernel Name) to accumulate per-invocation values, then sum
    # across all invocations for the whole-run totals.

    def safe_float(v):
        """Convert a metric value string like '246,922.75' to float, or NaN."""
        if pd.isna(v):
            return float("nan")
        s = str(v).replace(",", "").strip()
        if s in ("n/a", "N/A", ""):
            return float("nan")
        try:
            return float(s)
        except ValueError:
            return float("nan")

    df["value"] = df["Metric Value"].apply(safe_float)

    # Pivot: one row per kernel invocation, one column per metric
    pivot = df.pivot_table(
        index=["ID", "Kernel Name"],
        columns="Metric Name",
        values="value",
        aggfunc="first",
    ).reset_index()

    # Classify kernels
    conv_keywords = ["cudnn", "gemm", "conv", "fprop", "xmma", "cutlass", "sgemm"]
    def is_conv_kernel(name):
        low = name.lower()
        return any(k in low for k in conv_keywords)

    pivot["is_conv"] = pivot["Kernel Name"].apply(is_conv_kernel)

    # Metric helpers
    def col(metric):
        """Return the matching column name from the pivot, or None."""
        matches = [c for c in pivot.columns if metric in str(c)]
        return matches[0] if matches else None

    cycles_col = col("sm__cycles_active.avg")
    tensor_col = col("sm__pipe_tensor_op_hmma_cycles_active.sum")
    l1_col = col("l1tex__t_sectors.sum")
    l2_col = col("lts__t_sectors.sum")
    dram_r_col = col("dram__bytes_read.sum")
    dram_w_col = col("dram__bytes_write.sum")

    # Aggregate
    def total(column, subset=None):
        if column is None:
            return float("nan")
        series = pivot[column] if subset is None else pivot.loc[subset, column]
        return series.dropna().sum()

    conv_mask = pivot["is_conv"]

    result = {
        "total_kernels": len(pivot),
        "conv_kernels": int(conv_mask.sum()),
        # All kernels
        "total_cycles": total(cycles_col),
        "tensor_cycles": total(tensor_col),
        "l1_sectors": total(l1_col),
        "l2_sectors": total(l2_col),
        "dram_read_bytes": total(dram_r_col),
        "dram_write_bytes": total(dram_w_col),
        # Conv-only kernels (closest match to what SCALE-Sim models)
        "conv_cycles": total(cycles_col, conv_mask),
        "conv_tensor_cycles": total(tensor_col, conv_mask),
        "conv_l1_sectors": total(l1_col, conv_mask),
        "conv_l2_sectors": total(l2_col, conv_mask),
    }

    return result


# ---------------------------------------------------------------------------
# 2. Parse SCALE-Sim outputs
# ---------------------------------------------------------------------------
def parse_scalesim_outputs(report_dir="orin_nano_sim"):
    """Read the standard SCALE-Sim v2 report CSVs."""
    if not os.path.isdir(report_dir):
        print(f"  [!] SCALE-Sim output directory not found: {report_dir}")
        return None

    result = {}

    # COMPUTE_REPORT.csv
    compute_path = os.path.join(report_dir, "COMPUTE_REPORT.csv")
    if os.path.exists(compute_path):
        df = pd.read_csv(compute_path)
        df.columns = [c.strip() for c in df.columns]
        cycle_col = [c for c in df.columns if "Cycles" in c and "incl" in c.lower()]
        if not cycle_col:
            cycle_col = [c for c in df.columns if "Cycles" in c and "Stall" not in c]
        util_col = [c for c in df.columns if "Overall Util" in c]
        compute_util_col = [c for c in df.columns if "Compute Util" in c]
        stall_col = [c for c in df.columns if "Stall" in c]
        mapping_col = [c for c in df.columns if "Mapping" in c]

        result["total_cycles"] = int(df[cycle_col[0]].sum()) if cycle_col else 0
        result["overall_util"] = float(df[util_col[0]].mean()) if util_col else 0.0
        result["compute_util"] = float(df[compute_util_col[0]].mean()) if compute_util_col else 0.0
        result["total_stall_cycles"] = int(df[stall_col[0]].sum()) if stall_col else 0
        result["mapping_efficiency"] = float(df[mapping_col[0]].mean()) if mapping_col else 0.0
        result["num_layers"] = len(df)
        result["compute_df"] = df
    else:
        print(f"  [!] {compute_path} not found")

    # DETAILED_ACCESS_REPORT.csv
    access_path = os.path.join(report_dir, "DETAILED_ACCESS_REPORT.csv")
    if os.path.exists(access_path):
        df = pd.read_csv(access_path)
        df.columns = [c.strip() for c in df.columns]
        sram_read_cols = [c for c in df.columns if "SRAM" in c and "Reads" in c]
        sram_write_cols = [c for c in df.columns if "SRAM" in c and "Writes" in c]
        dram_read_cols = [c for c in df.columns if "DRAM" in c and "Reads" in c]
        dram_write_cols = [c for c in df.columns if "DRAM" in c and "Writes" in c]

        result["sram_reads"] = int(df[sram_read_cols].sum().sum()) if sram_read_cols else 0
        result["sram_writes"] = int(df[sram_write_cols].sum().sum()) if sram_write_cols else 0
        result["dram_reads"] = int(df[dram_read_cols].sum().sum()) if dram_read_cols else 0
        result["dram_writes"] = int(df[dram_write_cols].sum().sum()) if dram_write_cols else 0
        result["access_df"] = df

    # BANDWIDTH_REPORT.csv
    bw_path = os.path.join(report_dir, "BANDWIDTH_REPORT.csv")
    if os.path.exists(bw_path):
        df = pd.read_csv(bw_path)
        df.columns = [c.strip() for c in df.columns]
        result["bw_df"] = df

    return result


# ---------------------------------------------------------------------------
# 3. Format helpers
# ---------------------------------------------------------------------------
def fmt(v, unit=""):
    """Pretty-print a number."""
    if v is None or (isinstance(v, float) and (v != v)):  # NaN check
        return "N/A"
    if isinstance(v, str):  # If already a string, return as-is
        return v
    if isinstance(v, float):
        if v == int(v):
            return f"{int(v):,}{unit}"
        return f"{v:,.2f}{unit}"
    return f"{v:,}{unit}"


def pct_diff(hw, sim):
    """Percentage difference: (sim - hw) / hw * 100."""
    try:
        hw_f = float(hw)
        sim_f = float(sim)
        if hw_f == 0:
            return "N/A"
        return f"{((sim_f - hw_f) / hw_f) * 100:+.1f}%"
    except (TypeError, ValueError):
        return "N/A"


# ---------------------------------------------------------------------------
# 4. Main comparison report
# ---------------------------------------------------------------------------
def main():
    ncu_path = "inputs\\ncu_hardware_metrics.csv"
    scalesim_dir = "jetson_orin_nano_8gb"

    print("=" * 72)
    print("  Jetson Orin Nano — Hardware vs SCALE-Sim Comparison Report")
    print("=" * 72)

    # ---- Parse ----
    print("\n[1/3]  Parsing hardware metrics …")
    hw = parse_ncu_csv(ncu_path)

    print("[2/3]  Parsing SCALE-Sim results …")
    sim = parse_scalesim_outputs(scalesim_dir)

    if hw is None and sim is None:
        print("\n  ERROR: No data to compare. Run profiling and simulation first.")
        sys.exit(1)

    # ---- Hardware summary ----
    print("\n" + "-" * 72)
    print("  HARDWARE PROFILING SUMMARY  (Nsight Compute)")
    print("-" * 72)
    if hw:
        print(f"  Total GPU kernel invocations profiled : {hw['total_kernels']}")
        print(f"  Of which are conv/GEMM kernels        : {hw['conv_kernels']}")
        print(f"  Total SM cycles (all kernels)         : {fmt(hw['total_cycles'])}")
        print(f"  Conv-only SM cycles                   : {fmt(hw['conv_cycles'])}")
        print(f"  Tensor Core active cycles (all)       : {fmt(hw['tensor_cycles'])}")
        print(f"  Tensor Core active cycles (conv)      : {fmt(hw['conv_tensor_cycles'])}")
        print(f"  L1 cache traffic (all, sectors)       : {fmt(hw['l1_sectors'])}")
        print(f"  L2 cache traffic (all, sectors)       : {fmt(hw['l2_sectors'])}")
        print(f"  L1 cache traffic (conv, sectors)      : {fmt(hw['conv_l1_sectors'])}")
        print(f"  L2 cache traffic (conv, sectors)      : {fmt(hw['conv_l2_sectors'])}")
        dram_total = hw['dram_read_bytes'] + hw['dram_write_bytes']
        if dram_total != dram_total:  # NaN
            print(f"  DRAM traffic                          : N/A (unified memory)")
        else:
            print(f"  DRAM reads (bytes)                    : {fmt(hw['dram_read_bytes'])}")
            print(f"  DRAM writes (bytes)                   : {fmt(hw['dram_write_bytes'])}")
    else:
        print("  No hardware data available.")

    # ---- SCALE-Sim summary ----
    print("\n" + "-" * 72)
    print("  SCALE-Sim SIMULATION SUMMARY")
    print("-" * 72)
    if sim:
        print(f"  Layers simulated                      : {sim.get('num_layers', 'N/A')}")
        print(f"  Total cycles (incl. prefetch)         : {fmt(sim.get('total_cycles'))}")
        print(f"  Total stall cycles                    : {fmt(sim.get('total_stall_cycles'))}")
        print(f"  Average overall utilization           : {fmt(sim.get('overall_util'), '%')}")
        print(f"  Average compute utilization           : {fmt(sim.get('compute_util'), '%')}")
        print(f"  Average mapping efficiency            : {fmt(sim.get('mapping_efficiency'), '%')}")
        print(f"  Total SRAM reads (words)              : {fmt(sim.get('sram_reads'))}")
        print(f"  Total SRAM writes (words)             : {fmt(sim.get('sram_writes'))}")
        print(f"  Total DRAM reads (words)              : {fmt(sim.get('dram_reads'))}")
        print(f"  Total DRAM writes (words)             : {fmt(sim.get('dram_writes'))}")

        if "compute_df" in sim:
            print("\n  Per-layer breakdown:")
            cdf = sim["compute_df"]
            for _, row in cdf.iterrows():
                lid = int(row.iloc[0])
                cyc = int(row.iloc[1])
                util = float(row.iloc[4])
                print(f"    Layer {lid:2d}:  {cyc:>10,} cycles,  {util:5.1f}% util")
    else:
        print("  No simulation data available.")

    # ---- Side-by-side comparison ----
    print("\n" + "=" * 72)
    print("  SIDE-BY-SIDE COMPARISON")
    print("=" * 72)

    header = f"{'Metric':<36} {'Hardware (NCU)':>18} {'SCALE-Sim':>18} {'Δ':>10}"
    print(header)
    print("-" * 82)

    rows = []
    if hw and sim:
        # Cycles: use conv-only HW cycles vs SCALE-Sim total (SCALE-Sim only knows conv layers)
        rows.append((
            "Total Compute Cycles",
            hw.get("conv_cycles"),
            sim.get("total_cycles"),
        ))
        # On-chip memory
        hw_sram = hw.get("conv_l1_sectors", 0) + hw.get("conv_l2_sectors", 0)
        sim_sram = sim.get("sram_reads", 0) + sim.get("sram_writes", 0)
        rows.append((
            "On-chip Memory Access",
            hw_sram,
            sim_sram,
        ))
        # DRAM
        sim_dram = sim.get("dram_reads", 0) + sim.get("dram_writes", 0)
        rows.append((
            "Off-chip (DRAM) Access",
            "N/A (unified mem)" if (hw.get("dram_read_bytes", 0) != hw.get("dram_read_bytes", 0)) else hw.get("dram_read_bytes", 0) + hw.get("dram_write_bytes", 0),
            sim_dram,
        ))
        # Utilization / Tensor Core
        rows.append((
            "Tensor Util (cycles / %)",
            hw.get("conv_tensor_cycles"),
            f"{sim.get('overall_util', 0):.1f}%",
        ))

    for label, hw_val, sim_val in rows:
        delta = pct_diff(hw_val, sim_val) if not isinstance(hw_val, str) and not isinstance(sim_val, str) else "—"
        print(f"  {label:<34} {fmt(hw_val):>18} {fmt(sim_val):>18} {delta:>10}")

    # ---- Interpretation notes ----
    print("\n" + "-" * 72)
    print("  NOTES")
    print("-" * 72)
    print("""
  • SCALE-Sim models a pure weight-stationary systolic array.  The Jetson
    Orin Nano uses Ampere SMs with Tensor Cores, which have a fundamentally
    different microarchitecture.  Cycle counts are NOT directly comparable
    in absolute terms — compare RATIOS and TRENDS across layers instead.

  • 'On-chip Memory Access' maps HW L1+L2 cache sectors to SCALE-Sim SRAM
    word accesses.  Units differ (sectors=32 bytes vs. words=element size).

  • DRAM traffic on Jetson shows 'N/A' because the Orin Nano uses unified
    memory (no discrete DRAM bus that NCU can instrument in the same way).

  • Tensor Core cycles of 0 indicate PyTorch used the FP32/TF32 CUDA core
    path rather than the HMMA (half-precision matrix multiply) pipe.  To
    engage Tensor Cores, use FP16 inference or enable TF32 explicitly.
""")

    # ---- Save report to file ----
    report_path = "comparison_report.txt"
    # Re-run the print logic into a string
    buf = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = buf

    print("=" * 72)
    print("  Jetson Orin Nano - Hardware vs SCALE-Sim Comparison Report")
    print("=" * 72)
    print(f"\n  Hardware profiling: {ncu_path}")
    print(f"  Simulation results: {scalesim_dir}/")

    if hw:
        print(f"\n  --- Hardware (conv kernels) ---")
        print(f"  SM cycles        : {fmt(hw.get('conv_cycles'))}")
        print(f"  Tensor cycles    : {fmt(hw.get('conv_tensor_cycles'))}")
        print(f"  L1 sectors       : {fmt(hw.get('conv_l1_sectors'))}")
        print(f"  L2 sectors       : {fmt(hw.get('conv_l2_sectors'))}")

    if sim:
        print(f"\n  --- SCALE-Sim ---")
        print(f"  Total cycles     : {fmt(sim.get('total_cycles'))}")
        print(f"  Avg utilization  : {fmt(sim.get('overall_util'), '%')}")
        print(f"  SRAM reads       : {fmt(sim.get('sram_reads'))}")
        print(f"  SRAM writes      : {fmt(sim.get('sram_writes'))}")
        print(f"  DRAM reads       : {fmt(sim.get('dram_reads'))}")
        print(f"  DRAM writes      : {fmt(sim.get('dram_writes'))}")

    if hw and sim:
        print(f"\n  --- Comparison ---")
        for label, hw_val, sim_val in rows:
            delta = pct_diff(hw_val, sim_val) if not isinstance(hw_val, str) and not isinstance(sim_val, str) else "-"
            print(f"  {label:<34} {fmt(hw_val):>18} {fmt(sim_val):>18} {delta:>10}")

    sys.stdout = original_stdout
    report_text = buf.getvalue()

    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"  Report saved to: {report_path}")


if __name__ == "__main__":
    main()
