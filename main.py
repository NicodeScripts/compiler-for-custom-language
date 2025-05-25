import subprocess
import os
import sys
from ind_compiler import parse, lexer

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <source.ind>")
        return 1

    source_file = sys.argv[1]
    base_name = os.path.splitext(os.path.basename(source_file))[0]  # Extract filename without extension
    output_dir = base_name  # Create folder with the same name as the source file

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Compiling {source_file}...")

    # Step 1: Parse & Generate LLVM IR
    llvm_ir_file = os.path.join(output_dir, f"{base_name}.ll")
    if not parse(source_file, llvm_ir_file):
        print("Failed to generate LLVM IR. Compilation aborted.")
        return 1

    # Check if LLVM IR file was created
    if not os.path.exists(llvm_ir_file):
        print(f"Error: LLVM IR file '{llvm_ir_file}' was not created.")
        return 1

    print(f"✅ LLVM IR file generated: {llvm_ir_file}")

    # Step 2: Convert LLVM IR to Object Code
    obj_file = os.path.join(output_dir, f"{base_name}.o")
    try:
        print("Converting LLVM IR to object file...")
        subprocess.run(["llc", "-filetype=obj", llvm_ir_file, "-o", obj_file], check=True)
        print("✅ Object file created successfully.")
    except FileNotFoundError:
        print("❌ Error: LLVM compiler (llc) not found.")
        return 1
    except subprocess.CalledProcessError as e:
        print(f"❌ Error converting LLVM IR to object file:\n{e.stderr}")
        return 1

    # Step 3: Try to link & create executable
    executable = os.path.join(output_dir, f"out_{base_name}.ind")
    try:
        print(f"Linking object file to create executable...")
        subprocess.run(["clang", obj_file, "-o", executable], check=True)
        print(f"✅ Compilation successful! Run with: {executable}")
    except FileNotFoundError:
        print("❌ Error: Clang compiler not found.")
        return 1
    except subprocess.CalledProcessError as e:
        print(f"❌ Error linking object file:\n{e.stderr}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())

    # to compile:
    # python3 main.py basic.ind
    # run with ./out-basic.ind

#To-do:
# Add in strings
# Brackets for maths
# Functions?