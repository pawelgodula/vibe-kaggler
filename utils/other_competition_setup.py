# Utility function to set up a competition directory and fetch initial info.

"""
Extended Description:
Creates a directory structure for a new Kaggle competition under the `competition/` 
folder and fetches basic information (details, file list) using the Kaggle CLI via
the `get_kaggle_competition_info` utility.

The fetched information is saved into text files within the competition's directory.
"""

import os
from typing import Optional

# Import the helper function
try:
    from .get_kaggle_competition_info import get_kaggle_competition_info
except ImportError:
    # Allow running script directly for testing/dev
    from get_kaggle_competition_info import get_kaggle_competition_info

def competition_setup(competition_name: str, base_dir: str = "competition") -> None:
    """Sets up the directory and fetches initial data for a Kaggle competition.

    Args:
        competition_name (str): The name/slug of the Kaggle competition.
        base_dir (str, optional): The base directory where competition folders 
                                  are stored. Defaults to "competition".
    """
    if not competition_name:
        print("Error: Competition name cannot be empty.")
        return
        
    target_dir = os.path.join(base_dir, competition_name)
    details_file = os.path.join(target_dir, "competition_details.txt")
    files_file = os.path.join(target_dir, "competition_files.txt")

    print(f"Setting up directory for competition: '{competition_name}' in '{base_dir}/'")
    
    # 1. Create directory
    try:
        os.makedirs(target_dir, exist_ok=True)
        print(f"Directory created/exists: {target_dir}")
    except OSError as e:
        print(f"Error creating directory {target_dir}: {e}")
        return # Stop if directory creation fails

    # 2. Fetch info using the helper function
    print("\nFetching competition information using Kaggle CLI...")
    info = None
    try:
        info = get_kaggle_competition_info(competition_name)
    except FileNotFoundError:
        print("Operation failed: Kaggle CLI command ('kaggle') not found. Is it installed and in your PATH?")
        return
    except Exception as e:
        print(f"Operation failed: An unexpected error occurred while fetching info: {e}")
        return
        
    if not info:
         print("Failed to retrieve competition info.")
         return

    # Check for errors reported by the helper
    if info.get('error'):
        print(f"\nErrors encountered fetching competition info:")
        print(info['error'])
        # Decide whether to proceed saving partial info? Let's stop if main commands failed.
        print("Aborting setup due to errors fetching info.")
        return
    else:
         print("Successfully fetched competition info.")

    # 3. Save details
    if info.get('details'):
        try:
            with open(details_file, 'w', encoding='utf-8') as f:
                f.write(info['details'])
            print(f"Saved competition details to: {details_file}")
        except IOError as e:
            print(f"Error saving details to {details_file}: {e}")
    else:
        print("No competition details retrieved or saved.")

    # 4. Save files list
    if info.get('files'):
        try:
            with open(files_file, 'w', encoding='utf-8') as f:
                f.write(info['files'])
            print(f"Saved competition file list to: {files_file}")
        except IOError as e:
            print(f"Error saving file list to {files_file}: {e}")
    else:
         print("No competition file list retrieved or saved.")
         
    print(f"\nCompetition setup for '{competition_name}' finished.")

# Example Usage (if run as script)
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        comp_name = sys.argv[1]
    else:
        comp_name = 'titanic' # Default example
        print(f"No competition name provided, using default: '{comp_name}'")
        
    print(f"\n--- Running Competition Setup for: '{comp_name}' ---")
    competition_setup(comp_name)
    print("--- Setup Script Finished ---") 