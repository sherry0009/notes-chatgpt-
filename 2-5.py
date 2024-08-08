import requests
from bs4 import BeautifulSoup
import csv

def get_movie_reviews(movie_id, page_limit=10):
    url_template = f'https://movie.douban.com/subject/{movie_id}/comments?start={{start}}&limit=20&sort=new_score&status=P'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    reviews_data = []

    for i in range(page_limit):
        url = url_template.format(start=i * 20)
        response = requests.get(url, headers=headers)
        response.encoding = 'utf-8'

        soup = BeautifulSoup(response.text, 'html.parser')

        for item in soup.find_all('div', class_='comment-item'):
            user_element = item.find('span', class_='comment-info').find('a')
            user = user_element.text.strip()

            rating_element = item.find('span', class_='rating')
            rating = rating_element['title'] if rating_element else ''

            comment_element = item.find('span', class_='short')
            comment = comment_element.text.strip()

            reviews_data.append({
                'user': user,
                'rating': rating,
                'comment': comment,
            })

    return reviews_data

def save_to_csv(reviews_data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['user', 'rating', 'comment']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in reviews_data:
            writer.writerow(row)

movie_id = "1292722"  # 泰坦尼克号的电影ID
reviews_data = get_movie_reviews(movie_id)
save_to_csv(reviews_data, 'titanic_movie_reviews.csv')
print('Movie reviews data saved to titanic_movie_reviews.csv')
