import os
import pyperclip

# Ensure pyperclip uses the correct clipboard utility
pyperclip.set_clipboard("xclip")  # or "xsel" if you installed xsel

# Now copy the content
pyperclip.copy("Hello, clipboard!")
print("Text copied successfully.")


def get_all_py_files_content(directory, include_files=None, exclude_files=None, exclude_dirs=None):
    if exclude_files is None:
        exclude_files = []
    if exclude_dirs is None:
        exclude_dirs = []
    if include_files is None:
        include_files = []
    # Convert exclude_dirs to normalized paths relative to the base directory
    exclude_dirs_fullpath = set(os.path.normpath(os.path.join(directory, ex_dir)) for ex_dir in exclude_dirs)

    all_code = ""
    for root, dirs, files in os.walk(directory):
        # Filter out any directories in the current path that are in exclude_dirs_fullpath
        dirs[:] = [d for d in dirs if os.path.normpath(os.path.join(root, d)) not in exclude_dirs_fullpath]
        
        # Debug print statement to check excluded directories
        print(f"Currently processing directory: {root}")
        print(f"Remaining subdirectories to explore: {dirs}")
        
        for file in files:
            if (file.endswith('.py')  and file not in exclude_files) or (file in include_files):
                file_path = os.path.join(root, file)
                # Double-check if file path is within any excluded directory
                if any(ex_dir in file_path for ex_dir in exclude_dirs_fullpath):
                    print(f"Excluding file {file_path} because it is inside an excluded directory.")
                    continue
                # Get the relative path of the file from the base directory
                relative_path = os.path.relpath(file_path, directory)
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_code += f"\n# --- File: {file} ---\n"
                    all_code += f"# Relative Path: {relative_path}\n\n"
                    all_code += f.read()
                    all_code += f"\n# --- End of {file} ---\n"
    return all_code

# Get the current working directory
current_directory = os.getcwd()

# Specify the files and directories you want to exclude
exclude_files = ['paperclip.py', 'another_file_to_exclude.py']
exclude_dirs = ['myenv', 'faiss', 'mteb', 'db', 'mteb_dataset', 'mydb', 'results', 'results_overlap', 'sqlite-autoconf-3450200']
# include_files = ["Dockerfile", "deployment.yaml", "requirements.txt"]
include_files =[]
# Get the content of all .py files, excluding specified files and directories
code_content = get_all_py_files_content(current_directory, include_files, exclude_files, exclude_dirs)

# Copy the content to the clipboard
pyperclip.copy(code_content)

print("All Python code (excluding specified files and directories) has been copied to the clipboard.")
