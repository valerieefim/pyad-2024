import pickle
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Убедитесь, что загружены необходимые ресурсы
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")


def books_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Books.csv"""
    df = df.copy()
    # Удаление ненужных столбцов
    df.drop(columns=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], inplace=True)
    # Преобразование некорректных годов
    df['Year-Of-Publication'] = pd.to_numeric(df['Year-Of-Publication'], errors='coerce')
    df['Year-Of-Publication'] = df['Year-Of-Publication'].fillna(df['Year-Of-Publication'].median())
    return df


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.csv"""
    df = df.copy()
    # Исключаем нулевые рейтинги
    df = df[df['Book-Rating'] != 0]
    # Заменяем оценки на средние для каждой книги
    avg_ratings = df.groupby('ISBN')['Book-Rating'].mean()
    df['Book-Rating'] = df['ISBN'].map(avg_ratings)
    return df


def title_preprocessing(text: str) -> str:
    """Функция для нормализации текстовых данных в столбце Book-Title"""
    stop_words = set(stopwords.words("english"))
    text = re.sub(r'[^\w\s]', '', text)  # Удаляем пунктуацию
    tokens = nltk.word_tokenize(text.lower())  # Токенизация и приведение к нижнему регистру
    tokens = [word for word in tokens if word not in stop_words]  # Удаление стоп-слов
    return " ".join(tokens)


def modeling(books: pd.DataFrame, ratings: pd.DataFrame) -> None:
    """Функция для обучения линейной регрессии"""
    # Преобразование категориальных столбцов
    le_author = LabelEncoder()
    books['Book-Author'] = le_author.fit_transform(books['Book-Author'].fillna('Unknown'))

    le_publisher = LabelEncoder()
    books['Publisher'] = le_publisher.fit_transform(books['Publisher'].fillna('Unknown'))

    # Применение TF-IDF к названиям книг
    tfidf = TfidfVectorizer(max_features=1000)
    books['Book-Title'] = books['Book-Title'].apply(title_preprocessing)
    tfidf_matrix = tfidf.fit_transform(books['Book-Title']).toarray()

    # Объединение данных
    merged = ratings.merge(books, on='ISBN')
    features = ['Book-Author', 'Publisher', 'Year-Of-Publication']
    X = pd.concat([merged[features], pd.DataFrame(tfidf_matrix)], axis=1)
    y = merged['Book-Rating']

    # Загрузка тестового набора для выбора правильных признаков
    test_data = pd.read_csv("linreg_test.csv")
    test_features = test_data.columns.drop("y")  # Все столбцы, кроме целевой переменной

    # Убедимся, что X содержит только признаки, присутствующие в test_features
    X = X.reindex(columns=test_features, fill_value=0)

    # Масштабирование данных
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Сохранение масштабатора
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Разделение на тренировочную и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение модели
    linreg = SGDRegressor(random_state=42)
    linreg.fit(X_train, y_train)

    # Оценка модели
    y_pred = linreg.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error (MAE): {mae}")

    # Сохранение модели
    with open("linreg.pkl", "wb") as file:
        pickle.dump(linreg, file)

if __name__ == "__main__":
    # Загрузка данных
    books = pd.read_csv("Books.csv", low_memory=False)
    ratings = pd.read_csv("Ratings.csv")

    # Предобработка данных
    books = books_preprocessing(books)
    ratings = ratings_preprocessing(ratings)

    # Моделирование
    modeling(books, ratings)