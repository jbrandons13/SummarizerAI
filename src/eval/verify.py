import csv
from pathlib import Path
import sys

def verify_range(val_str: str, min_val: float, max_val: float, is_int: bool = False) -> bool:
    if val_str == "NaN":
        return True # NaN is tracked separately, doesn't break range check directly but is flagged
    try:
        val = float(val_str)
        if is_int:
            if not val.is_integer():
                return False
            val = int(val)
        return min_val <= val <= max_val
    except ValueError:
        return False

def main():
    eval_dir = Path("data/evaluation")
    files_to_check = [
        "m1_clipscore_per_group.csv",
        "m1_clipscore_per_video.csv",
        "m2_judge_visual.csv",
        "m3_judge_narrative.csv",
        "m4_summary_fidelity.csv",
        "summary_report.md"
    ]
    
    print("=== Verification Steps ===")
    
    # 1. File existence check
    all_exist = True
    print("\n1. File existence check:")
    for f in files_to_check:
        p = eval_dir / f
        if p.exists():
            print(f"  [PASS] {f} exists (size: {p.stat().st_size} bytes)")
        else:
            print(f"  [FAIL] {f} NOT FOUND")
            all_exist = False
            
    # 2. CSV sanity check
    print("\n2. CSV row count sanity:")
    csv_files = files_to_check[:-1]  # Exclude summary_report.md
    for f_name in csv_files:
        p = eval_dir / f_name
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            # Expecting 11 lines (header + 10 videos) for per-video files
            # m1_clipscore_per_group will have more lines.
            print(f"  {f_name}: {len(rows)} lines (including header)")
            
    # 3. Range checks
    print("\n3. Range and NaN checks:")
    range_passes = True
    
    # M1 Per Group
    m1_group_path = eval_dir / "m1_clipscore_per_group.csv"
    if m1_group_path.exists():
        with open(m1_group_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            nan_count = 0
            out_of_range = 0
            for row in reader:
                val = row["score"]
                if val == "NaN":
                    nan_count += 1
                elif not verify_range(val, 0.0, 2.5):
                    out_of_range += 1
            print(f"  m1_clipscore_per_group.csv score range [0.0, 2.5]: {'PASS' if out_of_range == 0 else 'FAIL'} ({out_of_range} out of range, {nan_count} NaNs)")
            if out_of_range > 0:
                range_passes = False
                
    # M1 Per Video
    m1_video_path = eval_dir / "m1_clipscore_per_video.csv"
    if m1_video_path.exists():
        with open(m1_video_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            nan_count = 0
            out_of_range = 0
            for row in reader:
                val = row["mean"]
                if val == "NaN":
                    nan_count += 1
                elif not verify_range(val, 0.0, 2.5):
                    out_of_range += 1
            print(f"  m1_clipscore_per_video.csv mean range [0.0, 2.5]: {'PASS' if out_of_range == 0 else 'FAIL'} ({out_of_range} out of range, {nan_count} NaNs)")
            if out_of_range > 0:
                range_passes = False
                
    # M2 Visual Judge
    m2_path = eval_dir / "m2_judge_visual.csv"
    if m2_path.exists():
        with open(m2_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            nan_counts = {"dim1": 0, "dim2": 0, "dim3": 0}
            out_of_range = 0
            for row in reader:
                for dim, col in [("dim1", "dim1_score"), ("dim2", "dim2_score"), ("dim3", "dim3_score")]:
                    val = row[col]
                    if val == "NaN":
                        nan_counts[dim] += 1
                    elif not verify_range(val, 1.0, 5.0, is_int=True):
                        out_of_range += 1
            print(f"  m2_judge_visual.csv score range [1, 5] (int): {'PASS' if out_of_range == 0 else 'FAIL'} ({out_of_range} out of range, NaNs: {nan_counts})")
            if out_of_range > 0:
                range_passes = False

    # M3 Narrative Judge
    m3_path = eval_dir / "m3_judge_narrative.csv"
    if m3_path.exists():
        with open(m3_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            nan_counts = {"dim1": 0, "dim2": 0, "dim3": 0}
            out_of_range = 0
            for row in reader:
                for dim, col in [("dim1", "dim1_score"), ("dim2", "dim2_score"), ("dim3", "dim3_score")]:
                    val = row[col]
                    if val == "NaN":
                        nan_counts[dim] += 1
                    elif not verify_range(val, 1.0, 5.0, is_int=True):
                        out_of_range += 1
            print(f"  m3_judge_narrative.csv score range [1, 5] (int): {'PASS' if out_of_range == 0 else 'FAIL'} ({out_of_range} out of range, NaNs: {nan_counts})")
            if out_of_range > 0:
                range_passes = False

    # M4 Summary Fidelity
    m4_path = eval_dir / "m4_summary_fidelity.csv"
    if m4_path.exists():
        with open(m4_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            nan_counts = {"r1": 0, "r2": 0, "rl": 0, "bs": 0}
            out_of_range = 0
            for row in reader:
                for col in ["rouge1_f1", "rouge2_f1", "rougeL_f1", "bertscore_f1"]:
                    val = row[col]
                    if val == "NaN":
                        nan_counts[col[:2]] += 1
                    elif not verify_range(val, 0.0, 1.0):
                        out_of_range += 1
            print(f"  m4_summary_fidelity.csv score range [0.0, 1.0]: {'PASS' if out_of_range == 0 else 'FAIL'} ({out_of_range} out of range, NaNs: {nan_counts})")
            if out_of_range > 0:
                range_passes = False

    # 4. Spot-check judge rationale
    print("\n4. Spot-check judge rationale (review_1):")
    # M2 Spot Check
    if m2_path.exists():
        with open(m2_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["video_id"] == "review_1":
                    print(f"  M2 (Visual) Rationale: {row['rationale']}")
                    break
    # M3 Spot Check
    if m3_path.exists():
        with open(m3_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["video_id"] == "review_1":
                    print(f"  M3 (Narrative) Rationale: {row['rationale']}")
                    break

    # 5. review_1 CLIPScore check
    print("\n5. review_1 CLIPScore value check:")
    if m1_video_path.exists():
        with open(m1_video_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["video_id"] == "review_1":
                    print(f"  review_1 CLIPScore mean: {row['mean']} (std: {row['std']})")
                    break

    print("\nVerification final status:")
    print(f"  File existence check: {'PASS' if all_exist else 'FAIL'}")
    print(f"  Range checks check: {'PASS' if range_passes else 'FAIL'}")

if __name__ == "__main__":
    main()
