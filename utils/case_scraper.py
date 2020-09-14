import re
import time
import requests
import numpy as np
import pandas as pd
import sqlite3
import bs4
from tqdm import tqdm


MONTHS = [
    'Jan/', 'Feb/', 'Mar/', 'Apr/', 'May/', 'Jun/', 'Jul/',
    'Aug/', 'Sep/', 'Oct/', 'Nov/', 'Dec/'
]

proxies = {
    'http' : 'http://206.189.157.23:80',
    'https' : 'https://206.189.157.23:80'
}

URL = 'http://elibrary.judiciary.gov.ph/thebookshelf/docmonth/'


def remove_html_tags(text):
    """Remove html tags from a string"""

    clean = re.compile(r'<.*?>|\[\d+\]')
    return re.sub(clean, '', text)


def get_cases(years, db, db_table, months=MONTHS):
    """Scrape the Supreme Court cases e-library by years"""

    cols = ["Case no.", "Title", "Date", "Link"]
    conn = sqlite3.connect(db)
    for year in tqdm(years):
        for month in months:
            ruling_page = f'{URL}{month}{year}'
            res_text = requests.get(ruling_page, proxies=proxies).text
            text_html = bs4.BeautifulSoup(res_text)
            cases = []
            div = 'div#container_title > ul > li'
            for item in text_html.select(div):
                cases.append(
                    [
                        item.a['href'].strip(),
                        item.a.strong.text.strip(),
                        item.a.small.text.strip(),
                        re.findall(r'\w+ \d{1,2}, \d{4}', item.a.text)[0]
                    ]
                )

            # save to db
            df_rulings = pd.DataFrame(cases, columns=cols)
            df_rulings.to_sql(db_table, con=conn, if_exists='append')
            time.sleep(np.random.random())


def get_case_details(case_links, db, db_table):
    """Get the details of the Supreme Court cases"""

    cols = ["Justice", "Text", "Link"]
    conn = sqlite3.connect(db)

    for case in tqdm(case_links):
        res_text = requests.get(case, proxies=proxies).text
        text_html = bs4.BeautifulSoup(res_text)
        case_details = []
        for content in text_html.select('div.single_content'):
            try:
                case_justice = content.p.strong.text
                case_text = content.select('div:nth-of-type(2)')[0]
                case_text = remove_html_tags(str(case_text).replace('<br/>', '\n'))
            except:
                case_justice = ''
                case_text = '\n\n'.join([remove_html_tags(str(x))
                                        for x in content.find_all('p')])
            case_details.append([case_justice, case_text, case])

        # save to db
        df_cases = pd.DataFrame(case_details, columns=cols)
        df_cases.to_sql(db_table, con=conn, if_exists='append')
        time.sleep(np.random.random())
