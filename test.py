import requests
from datetime import datetime

# Делаем GET-запрос к эндпоинту /post/recommendations/ для user_id=200
response = requests.get(
    "http://127.0.0.1:8000/post/recommendations/",
    params={"user_id": 256, "dt": datetime.now(), "limit": 3},
)

print("Рекомендации для пользователя c user_id = 256:")
print(response.json(), "\n")