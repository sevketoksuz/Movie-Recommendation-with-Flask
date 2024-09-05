from flask import Blueprint, render_template, request, redirect
from app.models import load_data, prepare_data, train_hybrid_model, get_top_n_recommendations
import pandas as pd
import time
import json

main = Blueprint('main', __name__)

@main.route('/')
def home():
    user_id = 611
    df_ratings, df_movies = load_data('./data/ratings.csv', './data/movies.csv')

    user_ratings = df_ratings[df_ratings['userId'] == user_id].merge(df_movies, on='movieId')

    return render_template('home.html', movies=df_movies.to_dict(orient='records'), user_ratings=user_ratings.to_dict(orient='records'))


@main.route('/rate', methods=['POST'])
def rate():
    user_id = 611  # Varsayılan kullanıcı ID
    rated_movies_json = request.form.get('ratedMovies')
    
    # Kullanıcı puanlarını veritabanına kaydetme
    df_ratings, df_movies = load_data('./data/ratings.csv', './data/movies.csv')
    
    # Kullanıcının zaten puanladığı filmleri kontrol et
    already_rated_movies = set(df_ratings[df_ratings['userId'] == user_id]['movieId'])

    if rated_movies_json:  # Eğer puanlanan filmler JSON verisi varsa
        try:
            rated_movies = json.loads(rated_movies_json)  # JSON stringi Python listesine çevir
            
            new_ratings = []
            for movie in rated_movies:
                movie_id = int(movie['movieId'])
                
                # Eğer film zaten puanlanmışsa, atla
                if movie_id in already_rated_movies:
                    continue
                
                rating = movie['rating']
                new_ratings.append({
                    'userId': user_id,
                    'movieId': movie_id,
                    'rating': float(rating),
                    'timestamp': int(time.time())  # Şu anki zamanı timestamp olarak ekleyin
                })
            
            # Eğer yeni derecelendirme eklenmemişse hata veya uyarı verebilirsiniz
            if not new_ratings:
                return "Hiçbir yeni film puanlanmadı veya tüm filmler zaten puanlanmış.", 400
            
            # Yeni derecelendirme verilerini DataFrame'e çevirin
            new_ratings_df = pd.DataFrame(new_ratings)

            # Eski ve yeni derecelendirme verilerini birleştirin
            df_ratings = pd.concat([df_ratings, new_ratings_df], ignore_index=True)

            # Eğitim veri setini güncelle
            df_ratings.to_csv('./data/ratings.csv', index=False)

            # Hibrit modeli eğit ve önerilerde bulun
            rating_matrix_scaled, df_movies_aligned, user_categories, item_categories = prepare_data(df_ratings, df_movies)
            W, H = train_hybrid_model(rating_matrix_scaled, df_movies_aligned)

            # Sabit kullanıcı ID ile öneriler al
            recommendations = get_top_n_recommendations(user_id, 5, W, H, user_categories, item_categories, df_movies, df_ratings)

            return render_template('recommendations.html', recommendations=recommendations.to_dict(orient='records'))
        
        except ValueError:
            return "Geçersiz değer girdiniz, lütfen tekrar deneyin.", 400
    else:
        return "Lütfen bir film ve puan seçiniz.", 400
    
@main.route('/update_rating', methods=['POST'])
def update_rating():
    user_id = 611
    movie_id = int(request.form.get('movieId'))
    new_rating = float(request.form.get('newRating'))
    
    df_ratings, df_movies = load_data('./data/ratings.csv', './data/movies.csv')
    
    # Güncelleme işlemi: ilgili satırı bulup puanı güncelleyin
    df_ratings.loc[(df_ratings['userId'] == user_id) & (df_ratings['movieId'] == movie_id), 'rating'] = new_rating
    df_ratings.to_csv('./data/ratings.csv', index=False)
    
    return redirect('/')

@main.route('/delete_rating', methods=['POST'])
def delete_rating():
    user_id = 611
    movie_id = int(request.form.get('movieId'))
    
    df_ratings, df_movies = load_data('./data/ratings.csv', './data/movies.csv')
    
    # Silme işlemi: ilgili satırı sil
    df_ratings = df_ratings[(df_ratings['userId'] != user_id) | (df_ratings['movieId'] != movie_id)]
    df_ratings.to_csv('./data/ratings.csv', index=False)
    
    return redirect('/')

@main.route('/get_recommendations', methods=['GET'])
def get_recommendations():
    user_id = 611  # Varsayılan kullanıcı ID
    df_ratings, df_movies = load_data('./data/ratings.csv', './data/movies.csv')
    
    # Öneri sistemi için gerekli hazırlıkları yapın
    rating_matrix_scaled, df_movies_aligned, user_categories, item_categories = prepare_data(df_ratings, df_movies)
    W, H = train_hybrid_model(rating_matrix_scaled, df_movies_aligned)

    # Sabit kullanıcı ID ile öneriler al
    recommendations = get_top_n_recommendations(user_id, 5, W, H, user_categories, item_categories, df_movies, df_ratings)

    return render_template('recommendations.html', recommendations=recommendations.to_dict(orient='records'))