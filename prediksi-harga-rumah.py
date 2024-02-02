import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder

st.set_page_config(layout='wide')

st.title('Dashboard untuk Prediksi Harga Rumah Menggunakan Machine Learning Linear Regression')

# Mengambil data
df = pd.read_csv('datafix_rumah.csv')
df['Provinsi'] = df['Provinsi'].replace('Unknown', 'Diluar Jawa dan Bali')

# Split data
# One-hot encoding pada kolom Provinsi
X = pd.get_dummies(df[['Kamar Tidur', 'Luas Bangunan', 'Luas Lahan', 'Provinsi']], drop_first=True)
y = df['Harga']
# st.dataframe(X)

tab1, tab2 = st.tabs(["Prediksi", "Konfigurasi"])

with tab2:
    # Split data into training set and test set
    test_size= st.slider('Porsi Data Uji', value= 0.2, min_value= 0.01, max_value= 0.99)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    # st.divider()
    # Linear Regression model
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # model evaluation on test data
    # create prediction vector on training data
    y_predict_test = model.predict(X_test)
    mae= mean_absolute_error(y_test, y_predict_test)
    mape= mean_absolute_percentage_error(y_test, y_predict_test) * 100
    st.text(f'MAE for test data is {mae:.2f}')
    st.text(f'MAPE for test data is {mape:.2f}%')

    import altair as alt

    # plot1, plot2 = st.columns(2)
    st.divider()
    # with plot1:
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    import altair as alt
    
    scatter_plot = alt.Chart(results).mark_circle(size=30).encode(
    x='Actual',
    y='Predicted',
    tooltip=['Actual', 'Predicted']
    ).interactive().properties(
    title='Actual vs Predicted Harga Rumah',
    width=700,
    height=400
    ).configure_title(
    align= 'center',
    fontSize=40
    )
    st.altair_chart(scatter_plot)

    # with plot2:
    le= LabelEncoder()
    df['Label Provinsi'] = le.fit_transform(df['Provinsi'])
    corr_df = df[['Kamar Tidur', 'Luas Bangunan', 'Luas Lahan', 'Label Provinsi', 'Harga']].corr()

    corr_df = corr_df.reset_index().melt('index')
    # corr_df
    corr_df.columns = ['Index', 'Variable', 'Correlation']
    # corr_df
    chart = alt.Chart(corr_df).mark_rect().encode(
    x='Index:O',
    y='Variable:O',
    color='Correlation:Q',
    tooltip=['Index', 'Variable', 'Correlation']
    ).properties(
        title='Korelasi antar Variabel',
        width=700,
        height=600
    ).configure_title(
    fontSize=40
    )
    st.altair_chart(chart)
with tab1:
    st.subheader('''
    Dashboard ini bertujuan untuk memprediksi harga rumah berdasarkan beberapa variabel yang diinputkan oleh Anda.
    Variabel yang digunakan dalam prediksi harga rumah ini adalah:
    - Jumlah kamar tidur
    - Luas bangunan
    - Luas lahan
    - Provinsi (Jawa, Bali, Luar Jawa dan Bali)
    
    Data yang digunakan dalam prediksi ini adalah data harga rumah dari seluruh Indonesia yang diambil dari website lamudi.co.id, dengan 1900+ baris data.
    ''')
    st.markdown('''    
    <div style="font-family: Poppins, sans-serif; color: red">
    Disclaimer: Data ini saya ambil secara manual dari scrapping website lamudi.co.id, dan data ini hanya digunakan untuk keperluan pembelajaran.
    </div>
    ''', unsafe_allow_html=True)
    st.divider()
    prediksi, tabel = st.columns(2)
    
    with prediksi:
        st.header('Prediksi Harga Rumah Anda')

        provinsi = st.selectbox('Wilayah (Provinsi)', df['Provinsi'].unique())
        kolom1, kolom2, kolom3 = st.columns(3)
        with kolom1:
            kamar_tidur = st.number_input('Jumlah Kamar Tidur', min_value=1, max_value=10, value=3)
            
        with kolom2:
            luas_bangunan = st.number_input('Luas Bangunan ($m^2$)', min_value=30, step=1, value= 50)
            
        with kolom3:
            luas_lahan = st.number_input('Luas Lahan ($m^2$)', min_value=35, step=1, value= 50)

        # membuat data dari input user
        data = pd.DataFrame([[kamar_tidur, luas_bangunan, luas_lahan, provinsi]], columns=['Kamar Tidur', 'Luas Bangunan', 'Luas Lahan', 'Provinsi'])
        data = pd.get_dummies(data)
        # st.dataframe(data)

        # menambah kolom agar sesuai dengan kolom training
        missing_cols = set(X.columns) - set(data.columns)
        for i in missing_cols:
            data[i] = 0
        data = data[X.columns]
        # st.dataframe(data)
        
        prediction = model.predict(data)
        tombol= st.button('Mulai Prediksi!', type='primary')
        if tombol== True:
            prediction_in_rupiah = "{:}".format(prediction[0])

    with tabel:
        def format_big_number(num):
            num = float(num)
            if num >= 1e9:
                return f"{num / 1e9:.2f} Miliar"
            elif num >= 1e6:
                return f"{num / 1e6:.2f} Juta"
            elif num >= 1e3:
                return f"{num / 1e3:.2f} K"
            else:
                return f"{num:.2f}"
            
        if tombol== False:
            st.write('')
        else:
            st.markdown(f'''
            <div style="
            font-size: 20px;
            font-family: poppins, sans-serif;
            color: white">
            Prediksi harga rumah Anda, dengan {kamar_tidur} kamar tidur, bangunan seluas {luas_bangunan}m&sup2;, lahan seluas {luas_lahan}m&sup2;, yang berada di Provinsi {provinsi} yaitu:
            ''', unsafe_allow_html=True)
            
            st.markdown(f'''
            <div style="
            font-size: 40px;
            font-family: Poppins, sans-serif;
            font-weight: bold;
            color: red">
                Rp {format_big_number(prediction_in_rupiah)}
            </div>
            ''', unsafe_allow_html=True)
