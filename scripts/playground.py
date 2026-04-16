from cliptools import FileProcessor
import glob
import os

# ClipData("scripts/testfile.clip")
# ClipData("scripts/template.clip")
# ClipData("scripts/prod.clip")
# ClipData("scripts/vector.clip")
# ClipData("scripts/svg_experiments/ver7.clip")
# ClipData("scripts/svg_experiments/ver7.clip")
# ClipData("scripts/svg_experiments/lots_of_points.clip")
# ClipData("scripts/svg_experiments/lots_of_points_512.clip")

# Setup CLI parsing
import argparse
parser = argparse.ArgumentParser(description="Process layer file.")
parser.add_argument("path", type=str, help="Path to file")
parser.add_argument("--find-psds", action="store_true", help="Find PSD files in folder")
args = parser.parse_args()

base_name = os.path.splitext(os.path.basename(args.path))[0]
temp_path = os.path.join("temp", base_name)
os.makedirs(temp_path, exist_ok=True)

if args.find_psds:
    # Find psds in all subfolders in the path
    psds = sorted(glob.glob(f"{args.path}/**/*.psd", recursive=True))
    for psd in psds:
        processor = FileProcessor(psd)
        folders = processor.export(path=temp_path, ext="png")
else:
    processor = FileProcessor(args.path)
    folders = processor.export(path=temp_path, ext="png")

print("Folders:", folders)