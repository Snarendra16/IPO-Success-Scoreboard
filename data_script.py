# script to get all ipo data from 


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

# ---- Step 1: Set up Chrome options ----
options = Options()
options.add_argument("--headless")          # run in background
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# ---- Step 2: Launch driver using webdriver_manager ----
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# ---- Step 3: Base URL ----
BASE_URL = "https://www.screener.in/screens/889928/ipo-last-3-years/?page=1"
driver.get(BASE_URL)
time.sleep(3)

# ---- Step 4: Auto-detect total pages ----
try:
    pagination_text = driver.find_element(By.CSS_SELECTOR, "div.flex.flex-between").text
    # Extract something like "Showing page 1 of 46"
    total_pages = int(pagination_text.split("of")[-1].strip().split()[0])
except Exception:
    total_pages = 46  # fallback
print(f"Detected total pages: {total_pages}")

# ---- Step 5: Loop through all pages ----
all_data = []
headers = []

for page in range(1, total_pages + 1):
    url = f"https://www.screener.in/screens/889928/ipo-last-3-years/?page={page}"
    driver.get(url)
    print(f"Scraping page {page}/{total_pages} ...")
    time.sleep(3)

    # Find table
    table = driver.find_element(By.CSS_SELECTOR, "table.data-table")
    rows = table.find_elements(By.TAG_NAME, "tr")

    # Extract headers (only once)
    if page == 1:
        headers = [th.text.strip() for th in rows[0].find_elements(By.TAG_NAME, "th")]

    # Extract rows
    for row in rows[1:]:
        cols = [td.text.strip() for td in row.find_elements(By.TAG_NAME, "td")]
        if cols:
            all_data.append(cols)

# ---- Step 6: Close driver ----
driver.quit()

# ---- Step 7: Convert to DataFrame and save ----
df = pd.DataFrame(all_data, columns=headers)
df.to_excel("IPO_Last_3_Years.xlsx", index=False)
df.to_csv("IPO_Last_3_Years.csv", index=False)

print(f"\n✅ Completed! Extracted {len(df)} IPO records across {total_pages} pages.")
