import os

def list_folders(directory):
    """
    List all folder names within the specified directory.
    """
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    return folders

def write_to_text_file(folders, filename):
    """
    Write folder names to a text file.
    """
    with open(filename, 'w') as file:
        for folder in folders:
            file.write(folder + '\n')

def main():
    # Specify the directory path
    directory = r"C:\Users\darkn\Vitamin Def\dataset\train"

    # Get the list of folder names
    folders = list_folders(directory)

    # Specify the text file name to write the folder names
    filename = 'folder_names.txt'

    # Write folder names to the text file
    write_to_text_file(folders, filename)

if __name__ == "__main__":
    main()
