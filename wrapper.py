import argparse
import subprocess
import sys
import os

# Removing generated files from previous execution

rotation_file_path = "./HybrIK/rotation_matrices_fencing.pt"
if os.path.exists(rotation_file_path):
    os.remove(rotation_file_path)
    print(f"Deleted: {rotation_file_path}")
else:
    print(f"No file to delete at: {rotation_file_path}")

translation_file_path = "./HybrIK/translation.pt"
if os.path.exists(translation_file_path):
    os.remove(translation_file_path)
    print(f"Deleted: {translation_file_path}")
else:
    print(f"No file to delete at: {translation_file_path}")

bones_file_path = "./bone_scale/bones.npz"
if os.path.exists(bones_file_path):
    os.remove(bones_file_path)
    print(f"Deleted: {bones_file_path}")
else:
    print(f"No file to delete at: {bones_file_path}")

offsets_file_path = "./body_model/scaled_offsets.npz"
if os.path.exists(offsets_file_path):
    os.remove(offsets_file_path)
    print(f"Deleted: {offsets_file_path}")
else:
    print(f"No file to delete at: {offsets_file_path}")

# Executing joint rotations and offsets file code

parser = argparse.ArgumentParser(description="Wrapper script for running demo_video_batched.py")
parser.add_argument('--filename', type=str, required=True, help='Path to the input video file')
args = parser.parse_args()

os.chdir("./HybrIK")

video_name = os.path.basename(args.filename)
video_stem = os.path.splitext(video_name)[0]
output_dir = f"res_{video_stem}"

subprocess.run([
    sys.executable, "scripts/demo_video_batched.py",
    "--video-name", args.filename,
    "--out-dir", output_dir,
    "--save-pk",
    "--save-img"
])
