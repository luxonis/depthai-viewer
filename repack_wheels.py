import os
import zipfile
import shutil
import argparse
import glob

def unpack_wheel(wheel_file, destination):
    """
    Unpack a .whl file to the specified destination directory.
    """
    if not os.path.exists(destination):
        os.makedirs(destination)
    print(f"Unpacking wheels to {destination}...")
    with zipfile.ZipFile(wheel_file, 'r') as zip_ref:
        zip_ref.extractall(destination)

    print(f"Unpacked {wheel_file} to {destination}")


def copy_packages_folder(source_folder, target_folder):
    """
    Copy the source folder into the target folder.
    """
    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"Source folder '{source_folder}' does not exist.")

    target_path = os.path.join(target_folder, os.path.basename(source_folder))

    if os.path.exists(target_path):
        shutil.rmtree(target_path)  # Remove existing folder if it exists

    shutil.copytree(source_folder, target_path)
    print(f"Copied {source_folder} to {target_path}")


def repack_wheel(source_folder, output_wheel):
    """
    Repack the folder into a .whl file.
    """
    with zipfile.ZipFile(output_wheel, 'w') as zip_ref:
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_folder)
                zip_ref.write(file_path, arcname)

    print(f"Repacked wheel file saved as {output_wheel}")


def main():
    # Paths (adjust these as necessary)
    parser = argparse.ArgumentParser(description="Repack a Python wheel file.")
    parser.add_argument("wheel_file", type=str, help="Path to the .whl file to be repacked.")
    args = parser.parse_args()

    # Locate .whl files
    wheel_files = glob.glob(os.path.join(args.wheel_file, "*.whl"))
    unpack_destination = "./unpacked_wheel"
    source_packages_folder = "./compilers"
    if not wheel_files:
        print("No .whl files found in the specified directory.")
        return

    for wheel_file in wheel_files:
        print(f"Processing: {wheel_file}")
        # Process each .whl file
        unpack_wheel(wheel_file, unpack_destination)
        backend_path = os.path.join(unpack_destination, "depthai_viewer", "_backend", "compilers")
        copy_packages_folder(source_packages_folder, backend_path)
        repack_wheel(unpack_destination, wheel_file)
        shutil.rmtree(unpack_destination)
        print("Temporary unpacked files removed.")


if __name__ == "__main__":
    main()
