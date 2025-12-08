import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
#import weather_data_processing

def preprocess_data(raw_df: pd.DataFrame):
    """
    Preprocess the raw dataframe.
    
    Args:
        raw_df (pd.DataFrame): The raw dataframe.
        
    Returns:
        Dict[str, Any]: Dictionary containing processed inputs and targets for train, val, and test sets.
    """
    input_cols = list(raw_df.columns)
    numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
    print(numeric_cols)
    categorical_cols = raw_df.select_dtypes('object').columns.tolist()
    imputer = joblib.load("model/imputer.pkl")
    raw_df[numeric_cols] = imputer.transform(raw_df[numeric_cols])
    scaler = joblib.load("model/scaler.pkl")
    raw_df[numeric_cols] = scaler.transform(raw_df[numeric_cols])
    ohe = joblib.load("model/ohe.pkl")
    encoded = ohe.transform(raw_df[categorical_cols])
    encoded_cols = list(ohe.get_feature_names_out(categorical_cols))
    raw_df = pd.concat([raw_df, pd.DataFrame(encoded, columns=encoded_cols, index=raw_df.index)], axis=1)
    raw_df.drop(columns=categorical_cols, inplace=True)

    return raw_df

def predict(raw_df):
    model = joblib.load('model/Weather.joblib')
    #data = np.expand_dims(np.array([sepal_l, sepal_w, petal_l, petal_w]), axis=0)
    data = preprocess_data(raw_df)
    predictions = model.predict(data)
    probability = model.predict_proba(data)
    return predictions[0], float(probability[0][1])

raw_df = {}
# Заголовок застосунку
st.image('img/Header_img.png')
st.title('Передбачення наявності опадів в Австралії')
st.markdown('Це проста модель для передбачення, чи піде завтра дощ в певному регіоні Австралії')

st.header("Введіть погодні умови")
col1, col2, col3 = st.columns(3)

# Введення характеристик чашолистків
with col1:
    raw_df['Location'] = st.selectbox(
        "Локація",
        ['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree', 'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond', 'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown', 'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat', 'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura', 'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns', 'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa', 'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport', 'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston', 'AliceSprings', 'Darwin', 'Katherine', 'Uluru'],
    )
    raw_df['MinTemp'] = st.slider('Мінімальна температура за добу', -8.5, 33.9)
    raw_df['MaxTemp'] = st.slider('Максимальна температура за добу', -4.8, 48.1)
    raw_df['Pressure9am'] = st.slider('Тиск о 9й годині ранку', 980.5, 1041.0)
    raw_df['Pressure3pm'] = st.slider('Тиск о 3й годині після обіду', 977.1, 1039.6)
    raw_df['WindSpeed9am'] = st.slider('Швидкість вітру о 9й годині ранку', 0.0, 130.0)
    raw_df['WindSpeed3pm'] = st.slider('Швидкість вітру о 3й годині після обіду', 0.0, 87.0)

# Введення характеристик пелюсток
with col2:
    raw_df['WindGustDir'] = st.selectbox(
        "Напрям найсильнішого пориву вітру",
        ['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE', 'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'],
    )    
    raw_df['WindGustSpeed'] = st.slider('Швидкість найсильнішого пориву вітру', 6.0, 135.0)
    raw_df['WindDir9am'] = st.selectbox(
        "Напрям вітру о 9й годині ранку",
        ['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE', 'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'],
    )
    raw_df['WindDir3pm'] = st.selectbox(
        "Напрям вітру о 3й годині після обіду",
        ['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE', 'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'],
    ) 
    raw_df['Humidity9am'] = st.slider('Вологість о 9й годині ранку', 0.0, 100.0)
    raw_df['Humidity3pm'] = st.slider('Вологість о 3й годині після обіду', 0.0, 100.0)
    raw_df['Sunshine'] = st.slider('Кількість сонячних годин', 0.0, 14.5)    

with col3:
    raw_df['Cloud9am'] = st.slider('Хмарність о 9й годині ранку', 980.5, 1041.0)
    raw_df['Cloud3pm'] = st.slider('Хмарність о 3й годині після обіду', 977.1, 1039.6)
    raw_df['Temp9am'] = st.slider('Температура о 9й годині ранку', 0.0, 130.0)
    raw_df['Temp3pm'] = st.slider('Температура о 3й годині після обіду', 0.0, 87.0)
    raw_df['Evaporation'] = st.slider('Випаровування води', 0.0, 14.5)    
    raw_df['Rainfall'] = st.slider('Кількість опадів за добу', 0.0, 14.5)    
    raw_df['RainToday'] = st.selectbox(
        "Чи був дощ сьогодні",
        ['Yes', 'No'],
    ) 
# Кнопка для прогнозування
raw_df = pd.DataFrame([raw_df])
new_order = ["Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustDir", "WindGustSpeed", "WindDir9am", "WindDir3pm", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm", "RainToday"]
raw_df = raw_df[new_order]

if st.button("Передбачення наявності дощу завтра"):
#    # Викликаємо функцію прогнозування
    print("6666666666666", raw_df.shape)
    result, proba = predict(raw_df)
    
    
    # обираємо фон
    if result == "Yes":
        bg_image_path = "img/Rain.png"
        label = "Завтра передбачається дощ"
        advice = "Не забудьте парасольку"
        probability = proba
    else:
        bg_image_path = "img/Sun.png"
        label = "Завтра не передбачається дощу"
        advice = "Можна планувати прогулянки"
        probability = 1 - proba

    def get_base64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    bg_image = get_base64(bg_image_path)
    
    # карточка передбачення
    st.markdown(
        f"""
        <div style="
            width: 100%;
            min-height: 200px;
            border-radius: 20px;
            background-image: url('data:image/png;base64,{bg_image}');
            background-size: cover;
            background-position: center;
            padding: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
        ">
            <div style="
                background: rgba(0,0,0,0.45);
                padding: 20px 30px;
                border-radius: 15px;
                display: flex;
                align-items: flex-start;
                gap: 20px;
                color: white;
            ">
                <div style="font-weight: 600; font-size: 22px; display:flex; align-items:center;">
                    Ваше пeредбачення:
                </div>
                <div style="display: flex; flex-direction: column; gap: 6px; max-width: 600px;">
                    <div style="font-size: 24px; font-weight: 700;">
                        {label}
                    </div>
                    <div style="font-size: 18px;">
                        Ймовірність передбачення: <b>{probability:.2f}</b>
                    </div>
                    <div style="font-size: 18px; opacity: 0.9;">
                        {advice}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
        <div style="background: rgba(0,0,0,0.05); padding: 15px 20px; border-radius: 12px; max-width: 600px;">
            <strong>Найважливіші ознаки:</strong>
            <ul style="margin-top: 8px; padding-left: 20px; line-height: 1.6; font-size: 18px;">
                <li>Вологість о 3й годині після обіду</li>
                <li>Швидкість найсильнішого пориву вітру</li>
                <li>Тиск о 3й годині після обіду</li>
                <li>Кількість сонячних годин</li>
                <li>Кількість опадів за добу</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


