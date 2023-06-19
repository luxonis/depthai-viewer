import argparse
import os
import re
import subprocess

resolved_paths = set()


def resolve_import(includes, path):
    shader_text = ""

    for include in includes:
        include_shader_path = include.replace("#import", "").replace("<", "").replace(">", "").strip()
        # Get absolute path to the included file
        include_file_path = os.path.abspath(os.path.join(os.path.dirname(path), include_shader_path))

        # Check if the file has already been included
        if include_file_path in resolved_paths:
            continue

        # Mark this file as included
        resolved_paths.add(include_file_path)

        with open(include_file_path, "r") as include_file:
            include_file_text = include_file.read()

        # Recursively resolve imports in the included file
        nested_includes = re.findall(r"#import <.*>", include_file_text)
        resolved_nested_includes = resolve_import(nested_includes, include_file_path)

        # Concatenate the resolved includes and the content of the included file
        shader_text += resolved_nested_includes + include_file_text

    return shader_text


# Parse which shader to check
parser = argparse.ArgumentParser(description="Check shader for errors.")
parser.add_argument("shader", metavar="shader", type=str, nargs=1, help="shader to check")

args = parser.parse_args()

shader = args.shader[0]

with open(shader, "r") as shader_file:
    shader_text = shader_file.read()


# Get all the imports: #import <path/to/file.wgsl>
includes = re.findall(r"#import <.*>", shader_text)
resolved_includes = resolve_import(includes, shader)

# Add the resolved includes to the original shader text
shader_text = resolved_includes + shader_text
includes = re.findall(r"#import <.*>", shader_text)
# Remove original import statements
for include in includes:
    shader_text = shader_text.replace(include, "")


subprocess.run(["naga", "--stdin-file-path", shader], input=shader_text.encode())
