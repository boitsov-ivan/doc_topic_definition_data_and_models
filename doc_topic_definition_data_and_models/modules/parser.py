import re
import time
import random
import os

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import csv

from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
import pickle


def edit(output_dir):
    def clean_hubs(hubs_string):
        if pd.isna(hubs_string):
            return ''
        
        hubs_list = str(hubs_string).split('|')
        
        cleaned_hubs = []
        for hub in hubs_list:
            hub = hub.strip().strip('*').strip()
            if hub and not hub.lower().startswith('блог'):
                cleaned_hubs.append(hub)
        return '|'.join(cleaned_hubs)


    def extract_article_id(url):
        """Извлекает числовой идентификатор из URL Habr"""
        match = re.search(r'/articles/(\d+)', str(url))
        if match:
            return int(match.group(1))
        return None
    
    def fix_hub_names(hub_list):
        fixed_list = []
        for hub in hub_list:
            if isinstance(hub, str):
                fixed_hub = re.sub(
                    r'Распредел.+ые системы',
                    'Распределённые системы',
                    hub
                )
                fixed_list.append(fixed_hub)
            else:
                fixed_list.append(hub)
        return fixed_list


    print(f"Cleaning data:")
    main_data_path = os.path.join(output_dir, "docs.csv")
    cleaned_data_path = os.path.join(output_dir, "docs_cleaned.csv")

    df = pd.read_csv(main_data_path, encoding="utf-8")
    print(f"Загружено статей: {len(df)}")

    
    initial_count = len(df)
    df = df[df['text'] != 'Без текста']
    removed_count = initial_count - len(df)
    print(f"Удалено статей без текста: {removed_count}")

    df['hubs'] = df['hubs'].apply(clean_hubs)
    df['tags'] = df['tags'].apply(clean_hubs)

    df['doc_id'] = df['url'].apply(extract_article_id)
    df = df.dropna(subset=['doc_id'])

    df['hubs'] = df['hubs'].apply(lambda x: x.split('|') if isinstance(x, str) else 'None')
    df = df[df['hubs'] != 'None']

    df['hubs'] = df['hubs'].apply(fix_hub_names)

    df['tags'] = df['tags'].apply(lambda x: x.split('|') if isinstance(x, str) else 'None')
    df = df[df['tags'] != 'None']

    df.to_csv(cleaned_data_path, index=False, encoding="utf-8")
    print(f"Очищенные данные сохранены в: {cleaned_data_path}")
    print(f"Итоговое количество статей: {len(df)}")

    return df



def run_browser():
    global driver
    try:
        options = Options()
        options.headless = True
        
        options.add_argument("--headless=new")

        options.add_argument("--disable-blink-features=AutomationControlled")
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(20)
        #print("Браузер успешно запущен!\n")
    except Exception as e:
        print(f"Ошибка запуска браузера: {e}\n")
        driver = None



def get_last_page_number(hub):
    """Ищет максимальный номер страницы со статьями по теме хаба"""

    url = f"https://habr.com/ru/hubs/{hub}/articles/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        page_links = soup.find_all('a', class_='tm-pagination__page')
        
        if not page_links:
            return 1
        
        max_page = 1
        for link in page_links:
            href = link.get('href', '')
            if '/page' in href:
                parts = href.split('/page')
                if len(parts) > 1:
                    num_part = parts[1].split('/')[0]
                    if num_part.isdigit():
                        page_num = int(num_part)
                        if page_num > max_page:
                            max_page = page_num
        
        return max_page
        
    except Exception:
        return 1



def get_pages(n: int, category: str) -> list:
    pages = []
    for i in range(1, n):
        pages.append(f'https://habr.com/ru/hubs/{category}/articles/page{i}/')
    return pages



def get_page_links(pages: list) -> list:
    links = []
    run_browser()
    for page in pages:
        try:
            driver.get(page)
            page1 = driver.page_source
            soup1 = BeautifulSoup(page1, 'lxml')
            links += ['https://habr.com' + el['href'] for el in soup1.find_all("a", class_="tm-title__link")]

            time.sleep(random.uniform(0.01, 0.03))
        except TimeoutException as ex:
            print("Exception has been thrown. " + str(ex))
    driver.quit()

    return links



def get_article(link: str, category: str) -> dict:
    article_dict = {}
    try:
        driver.get(link)
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'lxml')
        article_dict['category'] = category
        article_dict['url'] = link
        article_dict['summary'] = 'Без саммари'
        if soup.find('div', attrs={'id': 'post-content-body'}) is None:
            article_dict['text'] = 'Без текста'
        else:
            article_dict['text'] = soup.find('div', attrs={'id': 'post-content-body'}).text
        if soup.find('h1', class_='tm-title tm-title_h1') is None:
            article_dict['title'] = 'Без названия'
        else:
            article_dict['title'] = soup.find('h1', class_='tm-title tm-title_h1').text.replace("'", "")



        if soup.find_all('span', {'class': 'tm-publication-hub__link-container'}) is None:
            article_dict['hubs'] = 'Без хабов'
        else:
            article_dict['hubs'] = "|".join(list(map(lambda s: s.text, soup.find_all('span', {'class': 'tm-publication-hub__link-container'}))))
                                   

        article_dict['tags'] = "|".join([
            k.strip() 
            for k in soup.find('meta', {'name': 'keywords'})['content'].split(',')
        ]) if soup.find('meta', {'name': 'keywords'}) and soup.find('meta', {'name': 'keywords'}).get('content') else 'Без тегов'


        if soup.find("time")["title"] is None:
            article_dict['date'] = 'Без даты'
        else:
            article_dict['date'] = soup.find("time")["title"]
        #  ['category', 'url', 'title', 'text', 'summary', 'hubs', 'tags', 'date']
    except TimeoutException as ex:
        print("Exception has been thrown. " + str(ex))

    return article_dict



def parse_data(hubs, output_dir, max_links_pages_for_hub=1):
    print(f"Parsing data from hubs: {str(hubs)[1:-1]}.")
    
    main_data_path = os.path.join(output_dir, "docs.csv")
    progress_file = os.path.join(output_dir, "parsing_progress.pkl")
    
    if os.path.exists(main_data_path):
        try:
            existing_df = pd.read_csv(main_data_path, encoding="utf-8")
            existing_links = set(existing_df['link'].tolist())
            print(f"Загружено существующих данных: {len(existing_df)} статей")
            print(f"Уже обработано уникальных ссылок: {len(existing_links)}\n")
        except Exception as e:
            print(f"Ошибка загрузки существующих данных: {e}")
            existing_df = pd.DataFrame(columns=['category', 'url', 'title', 'text', 'summary', 'hubs', 'tags', 'date'])
            existing_links = set()
    else:
        existing_df = pd.DataFrame(columns=['category', 'url', 'title', 'text', 'summary', 'hubs', 'tags', 'date'])
        existing_links = set()
    
    
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'rb') as f:
                progress = pickle.load(f)
            
            last_hub = progress.get('last_hub', '')
            last_article = progress.get('last_article', 0)
            existing_df = progress.get('dataframe', existing_df)
            
            
            print(f"Восстанавливаем прогресс из {progress_file}\n")
            print(f"Последний обработанный хаб: {last_hub}\n")
            print(f"Обработано статей: {last_article}\n")
            print(f"Текущий размер датафрейма: {len(existing_df)} строк\n") 
            
        except Exception as e:
            print(f"Ошибка при загрузке прогресса: {e}\n")
            print("Начинаем новый парсинг\n")
            last_hub = ''
            last_article = 0
    else:
        print("Начинаем новый парсинг\n")
        last_hub = ''
        last_article = 0
    
    skip_hubs = bool(last_hub)
    
    for hub in tqdm(hubs, desc="ХАБЫ", unit="хаб", colour='green', position=0):
        if skip_hubs:
            if hub != last_hub:
                print(f"Пропускаем хаб '{hub}' (уже обработан)\n")
                continue
            else:
                skip_hubs = False
                print(f"Продолжаем с хаба '{hub}'\n")
        
        print(f"\n{'='*30}")
        print(f"Парсинг хаба: {hub}\n")
        print(f"{'='*30}")

        try:
            last_page = get_last_page_number(hub)
            print(f"Последняя страница хаба {hub}: {last_page}\n")
            print(f"Будет спарсено страниц: {last_page}\n")
            
            if max_links_pages_for_hub <= 0:
                max_links_pages_for_hub = last_page

            last_page = min(max_links_pages_for_hub, last_page)

            hub_pages = get_pages(last_page+1, hub)
            hub_links = get_page_links(hub_pages)
            
            print(f"Найдено ссылок на статьи: {len(hub_links)}\n")
            
            if not hub_links:
                print(f"Для хаба '{hub}' не найдено ссылок на статьи\n")
                continue
            
            new_hub_links = [link for link in hub_links if link not in existing_links]
            print(f"Новых ссылок для обработки: {len(new_hub_links)} из {len(hub_links)}\n")
            
            if not new_hub_links:
                print(f"Все ссылки для хаба '{hub}' уже обработаны, пропускаем\n")
                continue
            
            articles_pbar = tqdm(
                enumerate(new_hub_links),
                total=len(new_hub_links),
                desc=f"Статьи '{hub}'",
                unit="статья",
                colour='blue',
                leave=False
            )
        

            run_browser()
            for article_idx, article_link in articles_pbar:
                articles_pbar.set_postfix({
                    'статья': f"{article_idx + 1}/{len(new_hub_links)}",  
                    'всего': len(existing_df)  
                })
                
                try:
                    article_data = get_article(article_link, hub)
                    
                    if article_data:
                        new_row = pd.DataFrame([{
                            'category': hub,
                            'url': article_link,
                            'title': article_data.get('title', ''),
                            'text': article_data.get('text', ''),
                            'summary': article_data.get('summary', ''),
                            'hubs': article_data.get('hubs', ''),
                            'tags': article_data.get('tags', ''),
                            'date': article_data.get('date', '')
                        }])
                        
                        
                        existing_df = pd.concat([existing_df, new_row], ignore_index=True)
                        
                        
                    
                        existing_df.to_csv(main_data_path, index=False, encoding="utf-8")
                        
                        if (article_idx + 1) % 5 == 0:
                            progress_data = {
                                'last_hub': hub,
                                'last_article': article_idx,
                                'dataframe': existing_df,
                                'total_articles': len(new_hub_links),
                                'timestamp': time.time()
                            }
                            
                            with open(progress_file, 'wb') as f:
                                pickle.dump(progress_data, f)
                            
                            temp_path = os.path.join(output_dir, f"docs_{hub}_temp.csv")
                            existing_df.to_csv(temp_path, index=False, encoding="utf-8") 
                            
                            print(f"\nСохранен прогресс: статья {article_idx + 1}/{len(new_hub_links)}\n") 
                    
                    sleep_time = random.uniform(0.01, 0.02)
                    time.sleep(sleep_time)    
                    
                except Exception as e:
                    print(f"\nОшибка при парсинге статьи {article_link}: {e}\n")
                    
                    progress_data = {
                        'last_hub': hub,
                        'last_article': article_idx - 1 if article_idx > 0 else 0,
                        'dataframe': existing_df,
                        'error': str(e),
                        'failed_url': article_link,
                        'timestamp': time.time()
                    }
                    
                    with open(progress_file, 'wb') as f:
                        pickle.dump(progress_data, f)
                    
                    time.sleep(0.01)
                    continue
            
            articles_pbar.close()
            print(f"Хаб '{hub}' успешно обработан\n")
            
            # temp_hub_path = os.path.join(output_dir, f"docs_{hub}.csv")
            # hub_df = existing_df[existing_df['category'] == hub]
            # if len(hub_df) > 0:
            #     hub_df.to_csv(temp_hub_path, index=False, encoding="utf-8")
            #     print(f"Данные хаба '{hub}' сохранены ({len(hub_df)} статей)\n")
            
            driver.quit()
            
        except Exception as e:
            print(f"\nОшибка при обработке хаба '{hub}': {e}\n")
            import traceback
            traceback.print_exc()
            
            progress_data = {
                'last_hub': hub,
                'last_article': 0,
                'dataframe': existing_df,
                'error': str(e),
                'timestamp': time.time()
            }
            
            with open(progress_file, 'wb') as f:
                pickle.dump(progress_data, f)
            
            continue
    
    print(f"\n{'='*30}")
    print("Парсинг завершён!\n")
    print(f"Всего собрано статей: {len(existing_df)}\n")
    print(f"{'='*30}")
    
    print(f"Финальные данные сохранены в {main_data_path}\n")
    
    if os.path.exists(progress_file):
        os.remove(progress_file)
        print(f"Файл прогресса {progress_file} удалён\n")
    
    print("Очистка временных файлов...")
    for file in os.listdir(output_dir):
        if file.endswith("_temp.csv"):
            temp_path = os.path.join(output_dir, file)
            try:
                os.remove(temp_path)
                print(f"Удалён временный файл: {file}")
            except:
                pass
    
    return existing_df
