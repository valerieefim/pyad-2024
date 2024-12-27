import pandas as pd
import pickle
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.csv"""
    df = df.copy()
    df = df[df['Book-Rating'] != 0]  # Исключаем нулевые оценки

    # Исключаем книги, которые были оценены только один раз
    book_counts = df['ISBN'].value_counts()
    df = df[df['ISBN'].isin(book_counts[book_counts > 1].index)]

    # Исключаем пользователей, которые поставили только одну оценку
    user_counts = df['User-ID'].value_counts()
    df = df[df['User-ID'].isin(user_counts[user_counts > 1].index)]

    df.dropna(inplace=True)  # Удаляем строки с пропусками
    return df

from surprise.model_selection import GridSearchCV

def modeling(ratings: pd.DataFrame) -> None:
    """Функция для обучения модели SVD с оптимизацией"""
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)

    # Настройка гиперпараметров
    param_grid = {
        'n_factors': [50, 100, 200, 300],
        'reg_all': [0.01, 0.02, 0.05, 0.1],
        'lr_all': [0.001, 0.005, 0.01, 0.02],
    }
    gs = GridSearchCV(SVD, param_grid, measures=['mae'], cv=3)
    gs.fit(data)

    # Лучшая модель
    best_params = gs.best_params['mae']
    print(f"Лучшие параметры: {best_params}")
    svd = gs.best_estimator['mae']

    # Оценка на полной тренировочной выборке
    trainset = data.build_full_trainset()
    svd.fit(trainset)

    # Разделение на обучающую и тестовую выборки
    trainset, testset = train_test_split(data, test_size=0.2)

    # Тестирование модели
    predictions = svd.test(testset)
    mae = accuracy.mae(predictions)
    print(f"Mean Absolute Error (MAE): {mae}")

    # Сохранение модели
    with open("svd.pkl", "wb") as file:
        pickle.dump(svd, file)

if __name__ == "__main__":
    # Загрузка данных
    ratings = pd.read_csv("Ratings.csv")

    # Предобработка данных
    ratings = ratings_preprocessing(ratings)

    # Обучение модели
    modeling(ratings)