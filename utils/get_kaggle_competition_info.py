# Utility function to get information about a Kaggle competition using the CLI.

"""
Extended Description:
This function wraps the Kaggle command-line interface (CLI) to fetch details
and the file list for a specified competition. It requires the Kaggle CLI 
to be installed and authenticated (e.g., via kaggle.json).

It executes `kaggle competitions view -c <comp_name>` and 
`kaggle competitions files -c <comp_name>` using Python's subprocess module
and returns the captured output or error messages.
"""

import subprocess
import json # Potentially useful for parsing specific outputs if available
from typing import Dict, Any, Optional

def get_kaggle_competition_info(competition_name: str) -> Dict[str, Any]:
    """
    Fetches competition details and file list using the Kaggle CLI.

    Args:
        competition_name (str): The name/slug of the Kaggle competition 
                                (e.g., 'titanic', 'house-prices-advanced-regression-techniques').

    Returns:
        Dict[str, Any]: A dictionary containing information like:
                        'details': Raw output from 'kaggle competitions view'.
                        'files': Raw output from 'kaggle competitions files'.
                        'error': Error message if any command failed, None otherwise.
                        (Could be extended to parse specific fields if needed).
    
    Raises:
        FileNotFoundError: If the 'kaggle' command is not found (CLI not installed/in PATH).
    """
    results = {'details': None, 'files': None, 'error': None}
    error_messages = []
    
    try:
        # Get competition details/rules
        # Use '-q' for quieter output if needed, but view doesn't have it.
        cmd_details = ['kaggle', 'competitions', 'view', '-c', competition_name]
        print(f"Running command: {' '.join(cmd_details)}")
        process_details = subprocess.run(cmd_details, capture_output=True, text=True, check=False) # check=False to handle errors manually
        
        if process_details.returncode != 0:
            error_msg = f"Failed to get competition details. Error: {process_details.stderr.strip()}"
            print(f"Error: {error_msg}")
            error_messages.append(error_msg)
            # Decide if we should stop or try getting files anyway
            # For now, let's try getting files even if details fail
        else:
            results['details'] = process_details.stdout.strip()

        # Get competition files
        cmd_files = ['kaggle', 'competitions', 'files', '-c', competition_name]
        print(f"Running command: {' '.join(cmd_files)}")
        process_files = subprocess.run(cmd_files, capture_output=True, text=True, check=False)

        if process_files.returncode != 0:
            error_msg = f"Failed to get competition files. Error: {process_files.stderr.strip()}"
            print(f"Error: {error_msg}")
            error_messages.append(error_msg)
        else:    
            results['files'] = process_files.stdout.strip()

    except FileNotFoundError:
         # This happens if 'kaggle' command isn't found
         error_msg = "Kaggle CLI command ('kaggle') not found. Is it installed and in your PATH?"
         print(f"Error: {error_msg}")
         error_messages.append(error_msg)
         # Re-raise specific error for clarity
         raise FileNotFoundError(error_msg) 
    except Exception as e:
        # Catch other potential errors during subprocess execution
        error_msg = f"An unexpected error occurred during subprocess execution: {e}"
        print(f"Error: {error_msg}")
        error_messages.append(error_msg)
        # Optionally re-raise or just return the error in the dict
        
    if error_messages:
         results['error'] = "\n".join(error_messages)
         
    return results

# Example Usage (if run as script)
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        comp_name = sys.argv[1]
    else:
        comp_name = 'titanic' # Default example
        print(f"No competition name provided, using default: '{comp_name}'")
        
    print(f"\nAttempting to fetch info for competition: '{comp_name}'")
    
    try:
        info = get_kaggle_competition_info(comp_name)
        
        if info.get('details'):
            print(f"\n--- Details --- ")
            print(info['details'])
        else:
             print("\n--- Details: Failed to retrieve ---")
             
        if info.get('files'):
            print(f"\n--- Files --- ")
            print(info['files'])
        else:
             print("\n--- Files: Failed to retrieve ---")
             
        if info.get('error'):
            print(f"\n--- Errors Encountered --- ")
            print(info['error'])
            
    except FileNotFoundError as e:
         print(f"\nOperation failed: {e}")
    except Exception as e:
         print(f"\nAn unexpected error occurred: {e}") 