import io
import time
import pandas as pd
import streamlit as st

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person

st.title('顔診断アプリ')
st.header('画像から、性別、年齢、感情を分析します。')
st.subheader('画像をアップロードして下さい。')

# FaceAPIの設定
subscription_key = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxx' # AzureのAPIキー
endpoint = 'https://xxxxxxxxxxxxxxxxxxxxxxxxxxxxx/' # AzureのAPIエンドポイント

# クライアントを認証する
face_client = FaceClient(endpoint, CognitiveServicesCredentials(subscription_key))

status_text = st.empty()
# プログレスバー
progress_bar = st.progress(0)
for i in range(100):
        status_text.text(f'Progress: {i + 1}%')
        # for ループ内でプログレスバーの状態を更新する
        progress_bar.progress(i + 1)
        time.sleep(0.01)

# 検出した顔に描く長方形の座標を取得
def get_rectangle(faceDictionary):
    
    rect = faceDictionary.face_rectangle
    left = rect.left
    top = rect.top
    right = left + rect.width
    bottom = top + rect.height

    return (left, top, right, bottom)

# 描画するテキストを取得
def get_draw_text(faceDictionary, face_id):
    
    rect = faceDictionary.face_rectangle
    age = int(faceDictionary.face_attributes.age)
    gender = faceDictionary.face_attributes.gender
    emotion = faceDictionary.face_attributes.emotion
    text = f'{face_id} {gender} {age}'
    # 枠に合わせてフォントサイズを調整
    font_size = max(30, int(rect.width / len(text)))
    font = ImageFont.truetype(r'C:\windows\fonts\meiryo.ttc', font_size)

    return (text, font)
    
#感情パラメータ
def get_emotion(faceDictionary,face_id):
    
    global face_df
    emotion = faceDictionary.face_attributes.emotion
    anger = emotion.anger
    disgust = emotion.disgust
    fear = emotion.fear
    happiness = emotion.happiness
    neutral = emotion.neutral
    sadness = emotion.sadness
    surprise = emotion.surprise
    face_df[face_id]=[anger, disgust, fear, happiness, neutral, sadness, surprise]

# 認識された顔の上にテキストを描く座標を取得
def get_text_rectangle(faceDictionary, text, font):
    
    rect = faceDictionary.face_rectangle
    text_width, text_height = font.getsize(text)
    left = rect.left + rect.width / 2 - text_width / 2
    top = rect.top - text_height - 1

    return (left, top)

# テキストを描画
def draw_text(faceDictionary, face_id):

    text, font = get_draw_text(faceDictionary, face_id)
    text_rect = get_text_rectangle(faceDictionary, text, font)
    draw.text(text_rect, text, align='center', font=font, fill='red')
    
# アップロード    
uploaded_file = st.file_uploader("Choose an image...", type={"jpeg","jpg","png"})

if uploaded_file is not None:
    
    face_df = pd.DataFrame([],index=["怒気","嫌悪","恐怖","幸福","真顔","悲壮","驚き"])
    img = Image.open(uploaded_file)
    stream = io.BytesIO(uploaded_file.getvalue())

    detected_faces = face_client.face.detect_with_stream(
        stream, return_face_attributes=['age', 'gender','emotion'])

    if not detected_faces:
        raise Exception('画像から顔を検出できませんでした。')

    draw = ImageDraw.Draw(img)
    for face_id,face in enumerate(detected_faces):
        draw.rectangle(get_rectangle(face), outline='green', width=6)
        draw_text(face,face_id)
        get_emotion(face,face_id)
        print(face_df)

    st.image(img, caption='☆★☆感情分析の結果はこちら★☆★', use_column_width=True)
    st.subheader('感情パラメータ')
    st.table(face_df.style.highlight_max(axis=0))
    # エリアチャート
    st.subheader('感情グラフ')
    st.area_chart(face_df)
    
    st.balloons()
