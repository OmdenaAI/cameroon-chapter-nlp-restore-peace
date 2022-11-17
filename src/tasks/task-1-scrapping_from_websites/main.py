import asyncio
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd

async def savefile(title, date, author, content):

    df = pd.DataFrame({'Title': title, 'Pub Date': date, 'Author': author, 'Post Content': content})
    df.to_csv('Mimi mefo_posts.csv', header=False, mode='a')

async def scrape(Urls):
    path = 'F:/chromedriver_win32/chromedriver'
    driver = webdriver.Chrome(path)
    driver.get(Urls)

    # all_pages_tagged = driver.find_element(By.XPATH, "//a[@href='https://betatinz.com/tag/anglophone-crisis/']")
    # all_pages_tagged.click()

    posts = driver.find_elements(By.XPATH, "//div[@class='content-inner ']")
    headers = driver.find_elements(By.XPATH, "//div[@class='entry-header']")
    metas = driver.find_elements(By.XPATH, "//div[@class='meta_left']")

    title = []
    author = []
    date = []
    content = []

    for header in headers:
        t1 = header.find_element(By.XPATH, "./h1").text
        title.append(t1)
        print(t1)
    for meta in metas:
        a1 = meta.find_element(By.XPATH, "./div[@class='jeg_meta_author']").text
        author.append(a1)
        print(a1)
        d1 = meta.find_element(By.XPATH, "./div[@class='jeg_meta_date']/a").text
        date.append(d1)
        print(d1)

    for post in posts:
        c1 = post.find_element(By.XPATH, ".").text
        content.append(c1)
        print(c1)
        await savefile(title, date, author, content)


    driver.quit()


async def main():

    tasks = []
    with open('urls.csv') as file:
        csv_reader = csv.DictReader(file)
        for csv_row in csv_reader:
            print(csv_row['Urls'])
            task = asyncio.create_task(scrape(csv_row['Urls']))
            tasks.append(task)

loop = asyncio.get_event_loop()
loop.run_until_complete(main())