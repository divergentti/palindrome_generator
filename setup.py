from cx_Freeze import setup, Executable
import os

# Define the data folder
data_folder = 'data'

# Create a list of tuples for include_files
include_files = []
for dirpath, dirnames, filenames in os.walk(data_folder):
    for filename in filenames:
        file_path = os.path.join(dirpath, filename)
        # Calculate relative path
        relative_path = os.path.relpath(file_path, start=data_folder)
        include_files.append((file_path, os.path.join('data', relative_path)))

# Define build options
build_options = {
    'packages': [],
    'excludes': ['palindrome_word2vec.model',
        'palindrome_word2vec.model.wv.vectors_ngrams.npy'],
    'include_files': include_files  # Add the data files here
}

base = None  # Set this to 'Win32GUI' if you are building a GUI applicationls

executables = [
    Executable('PalindromiPeli.py', base=None)
]

setup(
    name='Palindrome_Game',
    version='0.32',
    description='Palindrome game',
    options={'build_exe': build_options},
    executables=executables
)
