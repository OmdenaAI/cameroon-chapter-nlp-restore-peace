from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd

website = 'https://mimimefoinfos.com/page/21?s=ambazonia'
path = 'F:/chromedriver_win32/chromedriver'
driver = webdriver.Chrome(path)
driver.get(website)

resultpage = driver.find_elements(By.XPATH, "//article[@class='jeg_post jeg_pl_lg_2 format-standard']/div/a")

url = []
for result in resultpage:
    l1 = result.find_element(By.XPATH, ".").get_attribute("href")
    url.append(l1)
    print(l1)

driver.quit()

df = pd.DataFrame({'Urls': url})
df.to_csv('Urls.csv', index=False)
print(df)