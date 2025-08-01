
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(layout="centered")
st.title("🔋 Previsão de Potência Eólica com Rede Neural")

uploaded_file = st.file_uploader("📤 Carregar arquivo CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Arquivo carregado com sucesso!")

        st.subheader("🔍 Pré-visualização dos Dados")
        st.write(df.head())

        required_cols = {'velocidade', 'direcao', 'altura', 'potencia'}
        if not required_cols.issubset(df.columns):
            st.error("❌ O arquivo precisa ter as colunas: velocidade, direcao, altura, potencia.")
            st.stop()

        # Pré-processamento
        X = df[['velocidade', 'direcao', 'altura']]
        y = df['potencia']

        scaler_X = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)

        scaler_y = MinMaxScaler()
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

        # Modelo
        model = Sequential([
            Dense(16, input_dim=3, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

        with st.spinner("🧠 Treinando modelo..."):
            history = model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), verbose=0)

        y_pred = scaler_y.inverse_transform(model.predict(X_test))
        y_test_inv = scaler_y.inverse_transform(y_test)

        # Métricas
        st.subheader("📊 Métricas")
        st.write(f"MAE: {mean_absolute_error(y_test_inv, y_pred):.2f} kW")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test_inv, y_pred)):.2f} kW")
        st.write(f"R²: {r2_score(y_test_inv, y_pred):.4f}")

        # Gráfico de dispersão
        st.subheader("📈 Potência Real vs. Prevista")
        fig1, ax1 = plt.subplots()
        ax1.scatter(y_test_inv, y_pred, color='blue')
        limite_max = max(np.max(y_test_inv), np.max(y_pred))
        ax1.plot([0, limite_max], [0, limite_max], '--', color='red')
        ax1.set_xlabel("Potência Real (kW)")
        ax1.set_ylabel("Potência Prevista (kW)")
        ax1.grid(True)
        st.pyplot(fig1)

        # Gráfico de loss
        st.subheader("📉 Curva de Erro")
        fig2, ax2 = plt.subplots()
        ax2.plot(history.history['loss'], label='Treinamento')
        ax2.plot(history.history['val_loss'], label='Validação')
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
else:
    st.info("⬆️ Faça upload de um arquivo CSV para começar.")
