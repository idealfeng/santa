import csv
import os
from pathlib import Path

OUTPUT_DIR = "runs/notebook_hunt_mother"
FINAL_SUBMISSION = "submission_mother_final.csv"

def generate_mother_submission(mother_csv, output_csv):
    print(f"Reading Mother Solution from {mother_csv}...")
    mother_200 = []

    # Standard python csv reading (no pandas)
    with open(mother_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract n from id 'n_i'
            row_id = row['id']
            n_str, i_str = row_id.split('_')
            if int(n_str) == 200:
                mother_200.append({
                    'i': int(i_str),
                    'x': row['x'],
                    'y': row['y'],
                    'deg': row['deg']
                })

    # Sort by 'i' to ensure correct order
    mother_200.sort(key=lambda x: x['i'])

    if len(mother_200) != 200:
        print(f"Warning: Expected 200 trees for N=200, found {len(mother_200)}")

    print("Generating prefixes for N=1..200...")

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'x', 'y', 'deg'])

        for n in range(1, 201):
            # Take the first n trees from the mother solution
            # This is the "prefix" strategy
            subset = mother_200[:n]

            for idx, tree in enumerate(subset):
                # Reconstruct ID as n_i
                new_id = f"{n}_{idx}"
                writer.writerow([new_id, tree['x'], tree['y'], tree['deg']])

    print(f"✅ Generated {output_csv} with 1..200 solutions.")

if __name__ == "__main__":
    # Check if post-opt file exists, otherwise use the ensemble file
    mother_source = Path(OUTPUT_DIR) / "ensemble_postopt.csv"
    if not mother_source.exists():
        print("Post-opt file not found, checking for ensemble...")
        mother_source = Path(OUTPUT_DIR) / "mother_200.csv"

    if not mother_source.exists():
        print(f"Error: Could not find strict mother source in {OUTPUT_DIR}")
        exit(1)

    output_file = f"{OUTPUT_DIR}/submission_raw_sliced.csv"
    generate_mother_submission(mother_source, output_file)
