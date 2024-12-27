import pickle
import pandas as pd
from surprise import SVD
from linreg_rec import books_preprocessing, ratings_preprocessing

def load_models():
    """Функция для загрузки моделей SVD и линейной регрессии."""
    with open("svd.pkl", "rb") as file:
        svd = pickle.load(file)

    with open("linreg.pkl", "rb") as file:
        linreg = pickle.load(file)

    return svd, linreg


def find_user_with_most_zeros(ratings: pd.DataFrame) -> int:
    """Функция для поиска пользователя с наибольшим количеством нулевых рейтингов."""
    zero_ratings = ratings[ratings["Book-Rating"] == 0]
    user_id = zero_ratings["User-ID"].value_counts().idxmax()
    return user_id


def make_recommendations(user_id: int, svd: SVD, linreg, books: pd.DataFrame, ratings: pd.DataFrame):
    """
    Функция для создания персональных рекомендаций:
    1. Находит книги с нулевыми рейтингами для пользователя.
    2. Использует модель SVD для предсказания рейтингов.
    3. Использует линейную регрессию для оценки средних рейтингов.
    4. Сортирует книги по убыванию среднего рейтинга.
    """
    zero_ratings = ratings[ratings["User-ID"] == user_id]
    zero_books = zero_ratings[zero_ratings["Book-Rating"] == 0]["ISBN"].unique()

    # Шаг 1: Предсказание рейтингов с помощью SVD
    svd_predictions = []
    for isbn in zero_books:
        pred = svd.predict(user_id, isbn).est
        if pred >= 8:  # Берем только книги с рейтингом >= 8
            svd_predictions.append((isbn, pred))

    if not svd_predictions:
        print("Нет рекомендаций для этого пользователя.")
        return

    # Шаг 2: Подготовка данных для линейной регрессии
    recommended_books = [isbn for isbn, _ in svd_predictions]
    book_features = books[books["ISBN"].isin(recommended_books)]

    # Подготовка признаков для линейной регрессии
    tfidf_vectorizer = pickle.load(open("tfidf.pkl", "rb"))
    book_features["Book-Title"] = book_features["Book-Title"].apply(title_preprocessing)
    tfidf_matrix = tfidf_vectorizer.transform(book_features["Book-Title"]).toarray()

    features = ["Book-Author", "Publisher", "Year-Of-Publication"]
    X = pd.concat([book_features[features].reset_index(drop=True), pd.DataFrame(tfidf_matrix)], axis=1)

    # Предсказание с помощью линейной регрессии
    linreg_predictions = linreg.predict(X)

    # Сортировка книг по линейной модели
    final_recommendations = sorted(
        zip(recommended_books, linreg_predictions),
        key=lambda x: x[1],
        reverse=True
    )

    # Печать рекомендаций
    print("Рекомендации для пользователя:")
    for isbn, score in final_recommendations:
        title = books.loc[books["ISBN"] == isbn, "Book-Title"].values[0]
        print(f"{title} (Рейтинг: {score:.2f})")


if __name__ == "__main__":
    # Загрузка данных
    books = pd.read_csv("Books.csv")
    ratings = pd.read_csv("Ratings.csv")

    # Предобработка данных
    books = books_preprocessing(books)
    ratings = ratings_preprocessing(ratings)

    # Загрузка моделей
    svd_model, linreg_model = load_models()

    # Поиск пользователя с наибольшим количеством нулевых рейтингов
    target_user = find_user_with_most_zeros(ratings)

    # Создание рекомендаций
    make_recommendations(target_user, svd_model, linreg_model, books, ratings)