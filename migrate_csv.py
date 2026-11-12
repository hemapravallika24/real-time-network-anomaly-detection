import csv
import os
import shutil

CSV_FILE = os.path.join("data", "network_events.csv")
BACKUP_FILE = os.path.join("data", "network_events_backup.csv")
TEMP_FILE = CSV_FILE + ".tmp"

EXPECTED_COLUMNS = [
    "timestamp",
    "packet_count",
    "bytes_transferred",
    "connection_duration",
    "failed_connections",
    "status",
    "anomaly_score",
    "risk_level",
    "reason"
]

if not os.path.exists(CSV_FILE):
    print("Error: CSV file not found.")
    raise SystemExit(1)

# Read the current CSV and preserve every valid row.
records = []

with open(CSV_FILE, "r", newline="", encoding="utf-8-sig") as file:
    reader = csv.reader(file)
    header = next(reader, None)

    if not header:
        print("Error: CSV file is empty.")
        raise SystemExit(1)

    header = [column.strip() for column in header]

    for line_number, row in enumerate(reader, start=2):
        if not row or all(not value.strip() for value in row):
            continue

        # Keep rows with either the old or new format.
        if len(row) > len(header):
            if len(row) != 9:
                print(f"Unexpected row format at line {line_number}.")
                raise SystemExit(1)

            # The original header has 7 fields, but this row has 9.
            row_header = EXPECTED_COLUMNS
        else:
            row_header = header

        record = dict(zip(row_header, row))

        # Add columns that did not exist in older records.
        for column in EXPECTED_COLUMNS:
            record.setdefault(column, "")

        # Historical records may not have these fields.
        if not record["risk_level"]:
            record["risk_level"] = "Unknown"

        if not record["reason"]:
            record["reason"] = "Not recorded by the earlier detector"

        records.append({
            column: record[column]
            for column in EXPECTED_COLUMNS
        })

# Create a backup of the current file only if one doesn't exist.
# Your existing backup remains untouched.
if not os.path.exists(BACKUP_FILE):
    shutil.copy2(CSV_FILE, BACKUP_FILE)

# Write the complete dataset to a temporary file first.
with open(TEMP_FILE, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=EXPECTED_COLUMNS)
    writer.writeheader()
    writer.writerows(records)

# Replace the original only after writing succeeds.
os.replace(TEMP_FILE, CSV_FILE)

print("\nCSV migration completed successfully!")
print(f"Total records preserved: {len(records)}")
print(f"Updated file: {CSV_FILE}")
print("\nLatest five records:")

for record in records[-5:]:
    print(record)