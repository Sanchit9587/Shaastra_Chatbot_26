import os

# The file where all code will be saved
OUTPUT_FILE = "codebase_context.txt"

# Extensions to include
INCLUDE_EXTS = {'.py', '.md'}

# Directories to ignore (Database files and cache)
IGNORE_DIRS = {
    'chroma_db', 
    'chroma_db_prod', 
    '__pycache__', 
    'venv', 
    '.git'
}

def export_codebase():
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        # Walk through the current directory
        for root, dirs, files in os.walk("."):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in files:
                # Check file extension
                _, ext = os.path.splitext(file)
                
                # Skip this script itself and the output file
                if file == "export_code.py" or file == OUTPUT_FILE:
                    continue
                
                if ext in INCLUDE_EXTS:
                    filepath = os.path.join(root, file)
                    
                    # Create a clear header for each file
                    header = f"\n{'='*50}\nFILE: {filepath}\n{'='*50}\n"
                    outfile.write(header)
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8') as infile:
                            content = infile.read()
                            outfile.write(content)
                            outfile.write("\n") # Ensure spacing between files
                    except Exception as e:
                        outfile.write(f"\n# Error reading file: {e}\n")
                        print(f"Skipping {file}: {e}")

    print(f"✅ Successfully exported code to: {OUTPUT_FILE}")

if __name__ == "__main__":
    export_codebase()