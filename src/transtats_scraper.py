#!python3
# -*- coding: utf-8 -*-
"""
@author: Micah Borrero
"""
# %%
import os
import time
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support.ui import Select
import zipfile

dat_select = {"2023": ["January"]}

timeout = 120

# Set up Edge options
download_dir = os.path.abspath("../data/ot_reporting")
edge_options = Options()
edge_options.add_experimental_option("prefs", {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
})

# Set up the Edge WebDriver
service = Service('../src/msedgedriver.exe')
driver = webdriver.Edge(service=service, options=edge_options)

# Navigate to the page
# https://transtats.bts.gov/PREZIP/ alternative download link
url = "https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr"
driver.get(url)

def check_new_zip(download_dir, start_time):
    """
    Get the newest ZIP file downloaded since the given start time.

    Args:
        download_dir (str): The directory to monitor.
        start_time (float): The timestamp when the monitoring started.

    Returns:
        str: The path to the newest ZIP file, or None if no valid file is found.
    """
    # Filter ZIP files created after the start time
    zip_files = [
        os.path.join(download_dir, f)
        for f in os.listdir(download_dir)
        if f.endswith('.zip') and os.path.isfile(os.path.join(download_dir, f)) 
        and os.path.getctime(os.path.join(download_dir, f)) > start_time
    ]

    if not zip_files:
        return None

    # Return the newest ZIP file by creation time
    newest_zip = max(zip_files, key=os.path.getctime)
    return newest_zip


def extract_zip(zip_path):
    """
    Extracts CSV files from the specified ZIP file and deletes the ZIP afterward.
    
    Args:
        zip_file_path (str): Path to the ZIP file.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file not found at {zip_path}")
    
    try:
        # Extract files
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List all files in the ZIP
            file_list = zip_ref.namelist()
            print(f"Files in ZIP: {file_list}")
            
            # Filter for CSV files
            csv_files = [f for f in file_list if f.endswith('.csv')]
            
            if not csv_files:
                print("No CSV files found in the ZIP.")
                return
            
            # Extract all CSV files
            for csv_file in csv_files:
                zip_ref.extract(csv_file, os.path.dirname(zip_path))
                print(f"Extracted: {csv_file}")
        
        # Delete the ZIP file
        os.remove(zip_path)
        print(f"Deleted ZIP file: {zip_path}")
    
    except zipfile.BadZipFile:
        print("Error: The file is not a valid ZIP archive.")
    except Exception as e:
        print(f"An error occurred: {e}")

for y in dat_select:
    for m in dat_select[y]:
        # Reset the start time for each download
        start_time = time.time()

        # Select the year from the dropdown
        year_dropdown = driver.find_element("id", "cboYear")
        select = Select(year_dropdown)
        select.select_by_visible_text(y)

        # Select the month from the dropdown
        month_dropdown = driver.find_element("id", "cboPeriod")
        select = Select(month_dropdown)
        select.select_by_visible_text(m)

        # Ensure all checkboxes are checked
        """
        # Alternate method to check all possible checkboxes
        checkboxes = driver.find_elements("xpath", '//input[@type="checkbox"]')
        for i in range(len(checkboxes)):
            checkbox = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, f'(//input[@type="checkbox"])[{i+1}]'))
            )
            if not checkbox.is_selected():
                driver.execute_script("arguments[0].click();", checkbox)
        """
        checkbox = driver.find_element("xpath", '//input[@name="chkAllVars"]')
        if not checkbox.is_selected():
            try:
                checkbox.click()
            except Exception:
                # Fallback to JavaScript if normal click fails
                driver.execute_script("arguments[0].click();", checkbox)

        # Check the ZIP download option as it is the most efficient
        checkbox = driver.find_element("xpath", '//input[@name="chkDownloadZip"]')
        if not checkbox.is_selected():
            try:
                checkbox.click()
            except Exception:
                # Fallback to JavaScript if normal click fails
                driver.execute_script("arguments[0].click();", checkbox)


        # Trigger the download
        download_button = driver.find_element("xpath", '//input[@type="submit" and @value="Download"]')
        download_button.click()

        print("Monitoring for new ZIP files...")
        try:
            while True:
                newest_zip = check_new_zip(download_dir, start_time)
                if newest_zip:
                    print(f"Newest ZIP file found: {newest_zip}")
                    break
                time.sleep(2)

                if time.time() - start_time > timeout:
                    raise TimeoutError("No new ZIP file was found within the timeout period.")

        except TimeoutError as e:
            print(e)

        # Extract the ZIP file
        extract_zip(newest_zip)

driver.quit()