import os
import sys
import glob
import shutil
import chromedriver_autoinstaller



def setup_chrome_driver() -> str:
    """
    Ensures ChromeDriver is installed and returns its path.
    - Checks for an existing compatible ChromeDriver.
    - If not found, installs the latest compatible version.
    """
    # Identify correct ChromeDriver filename based on OS
    driver_filename = "chromedriver.exe" if sys.platform in ["win32", "cygwin"] else "chromedriver"

    # Search for ChromeDriver recursively in the current directory
    glob_result = glob.glob(os.path.join(os.getcwd(), "**", driver_filename), recursive=True)

    if glob_result:  # If ChromeDriver is found, return the first match
        return os.path.abspath(glob_result[0])

    # Install the correct ChromeDriver version
    driver_path = chromedriver_autoinstaller.install()

    # Ensure the installed path is absolute
    return os.path.abspath(driver_path)

def remove_chromedriver(chrome_driver_path):
    """
    Removes the ChromeDriver after execution.
    """
    directory_path = os.path.dirname(chrome_driver_path)
    try:
        shutil.rmtree(directory_path)
    except Exception:
        print(f"Please remove ChromeDriver located at {directory_path} manually after process completion.")
