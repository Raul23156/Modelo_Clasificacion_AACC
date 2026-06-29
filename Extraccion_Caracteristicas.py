import os
import numpy as np
import mne
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from scipy.integrate import simpson


## Funciones auxiliares
def entropia_de_shannon(signal, bins=50):
    hist, _ = np.histogram(signal, bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


def caracteristicas_espectrales(signal_band, signal_full, sfreq):
    freqs_band, psd_band = welch(signal_band, sfreq, nperseg=sfreq*2)
    freqs_full, psd_full = welch(signal_full, sfreq, nperseg=sfreq*2)
    
    # Potencia absoluta de la banda seleccionada
    welch_abs = simpson(psd_band, freqs_band)   
    
    # Potencia absoluta de la banda completa (1-40 Hz)
    welch_abs_full = simpson(psd_full, freqs_full)

    # Potencia relativa intra-canal
    welch_rel_intra = welch_abs/welch_abs_full
  
    return welch_abs, welch_rel_intra


## Función principal para extraer caracteristicas por canal
def extraer_caracteristicas(banda, banda_completa):
    
    raw_band = mne.io.read_raw_fif(banda, preload=True)
    data_band = raw_band.get_data()
    
    raw_full = mne.io.read_raw_fif(banda_completa, preload=True)
    data_full = raw_full.get_data()

    ch_names = raw_band.ch_names
    sfreq = raw_band.info['sfreq']
    
    features = []           # Lista de características
    absolute_powers = []    # Lista de potencias absolutas de los canales

    for i, ch in enumerate(ch_names):
        signal_band = data_band[i]
        signal_full = data_full[i]

        mean = np.mean(signal_band)                     # Media
        std = np.std(signal_band)                       # Desviación estandar
        var = np.var(signal_band)                       # Varianza
        skewness = skew(signal_band)                    # Asimetría
        kurt = kurtosis(signal_band)                    # Curtosis
        rms = np.sqrt(np.mean(signal_band**2))          # Valor cuadrático medio

        primera_derivada = np.diff(signal_band)         # Primera derivada
        segunda_derivada = np.diff(primera_derivada)    # Segunda derivada

        mean_d1 = np.mean(primera_derivada)             # Media de la primera derivada
        std_d1 = np.std(primera_derivada)               # Desviacion estandar de la primera derivada

        mean_d2 = np.mean(segunda_derivada)             # Media de la segunda derivada
        std_d2 = np.std(segunda_derivada)               # Desviacion estandar de la segunda derivada

        peak_to_peak = np.ptp(signal_band)              # Rango
        max_val = np.max(signal_band)                   # Máximo
        min_val = np.min(signal_band)                   # Mínimo

        integral_signal = simpson(signal_band)          # Integral

        entropy = entropia_de_shannon(signal_band)      # Entropía

        # Espectrales (Potencia absoluta y Potencia relativa intra-canal)
        welch_abs, welch_rel_intra = caracteristicas_espectrales(signal_band, signal_full, sfreq)

        absolute_powers.append(welch_abs)   # Lista de potencias absolutas de los canales

        # Guardar características
        features.append({
            "canal": ch,
            "media": mean,
            "desviacion_estandar": std,
            "varianza": var,
            "asimetria": skewness,
            "curtosis": kurt,
            "valor_cuadratico_medio": rms,
            "rango": peak_to_peak,
            "maximo": max_val,
            "minimo": min_val,
            "media_d1": mean_d1,
            "desviacion_estandar_d1": std_d1,
            "media_d2": mean_d2,
            "desviacion_estandar_d2": std_d2,
            "integral": integral_signal,
            "entropia": entropy,
            "potencia_absoluta": welch_abs,
            "potencia_relativa_intra-canal": welch_rel_intra,
        })

    global_power = np.sum(absolute_powers)  # Suma de potencias absolutas de todos los canales
    for idx, feature_dict in enumerate(features):
        # Potencia relativa entre canales
        feature_dict["potencia_relativa"] = (absolute_powers[idx] / global_power if global_power != 0 else 0)

    df = pd.DataFrame(features)

    feature_vector = []     # Vector de características (filas de la matriz de características)

    # Recorrer canales fila por fila
    for _, row in df.iterrows():
        values = row.drop(labels=["canal"]).values.tolist() # Eliminar nombre del canal
        feature_vector.extend(values)                       # Añadir características al vector


    feature_names = []
    for ch in df["canal"]:
        for col in df.columns[1:]:
            feature_names.append(f"{ch}_{col}")


    return feature_vector, feature_names


## Programa MAIN
# Matrices de caracteristicas
filas_matriz_caracteristicas_delta = []
filas_matriz_caracteristicas_theta = []
filas_matriz_caracteristicas_alpha = []
filas_matriz_caracteristicas_beta = []
filas_matriz_caracteristicas_gamma = []
filas_matriz_caracteristicas_full = []

# Carga de archivos
ruta_archivos_preprocesados = os.path.join(os.getcwd(), "EEGs", "EEG_preprocesado")

# Recorrer todos los elementos dentro de la carpeta principal
for colegio in os.listdir(ruta_archivos_preprocesados):                             # Recorro carpetas de colegios
    ruta_colegio = os.path.join(ruta_archivos_preprocesados, colegio)
    for curso in os.listdir(ruta_colegio):                                          # Recorro carpetas de cursos
        ruta_curso = os.path.join(ruta_colegio, curso, "EEG")
        for iniciales in os.listdir(ruta_curso):                                    # Recorro carpetas de iniciales
            ruta_iniciales = os.path.join(ruta_curso, iniciales)

            ruta_sesion = os.path.join(ruta_iniciales, os.listdir(ruta_iniciales)[0])

            banda_delta = os.path.join(ruta_sesion, os.listdir(ruta_sesion)[0], "banda_delta.fif")
            banda_theta = os.path.join(ruta_sesion, os.listdir(ruta_sesion)[0], "banda_theta.fif")
            banda_alpha = os.path.join(ruta_sesion, os.listdir(ruta_sesion)[0], "banda_alpha.fif")
            banda_beta = os.path.join(ruta_sesion, os.listdir(ruta_sesion)[0], "banda_beta.fif")
            banda_gamma = os.path.join(ruta_sesion, os.listdir(ruta_sesion)[0], "banda_gamma.fif")
            banda_completa = os.path.join(ruta_sesion, os.listdir(ruta_sesion)[0], "banda_full.fif")

            vector_caracteristicas_delta, lista_caracteristicas = extraer_caracteristicas(banda_delta, banda_completa)
            vector_caracteristicas_theta, lista_caracteristicas = extraer_caracteristicas(banda_theta, banda_completa)
            vector_caracteristicas_alpha, lista_caracteristicas = extraer_caracteristicas(banda_alpha, banda_completa)
            vector_caracteristicas_beta, lista_caracteristicas = extraer_caracteristicas(banda_beta, banda_completa)
            vector_caracteristicas_gamma, lista_caracteristicas = extraer_caracteristicas(banda_gamma, banda_completa)
            vector_caracteristicas_full, lista_caracteristicas = extraer_caracteristicas(banda_completa, banda_completa)

            primera_columna = iniciales + "_" + colegio + "_" + curso
            vector_caracteristicas_delta.insert(0, primera_columna)
            vector_caracteristicas_theta.insert(0, primera_columna)
            vector_caracteristicas_alpha.insert(0, primera_columna)
            vector_caracteristicas_beta.insert(0, primera_columna)
            vector_caracteristicas_gamma.insert(0, primera_columna)
            vector_caracteristicas_full.insert(0, primera_columna)

            filas_matriz_caracteristicas_delta.append(vector_caracteristicas_delta)
            filas_matriz_caracteristicas_theta.append(vector_caracteristicas_theta)
            filas_matriz_caracteristicas_alpha.append(vector_caracteristicas_alpha)
            filas_matriz_caracteristicas_beta.append(vector_caracteristicas_beta)
            filas_matriz_caracteristicas_gamma.append(vector_caracteristicas_gamma)
            filas_matriz_caracteristicas_full.append(vector_caracteristicas_full)


lista_caracteristicas.insert(0, "ID")            

# Matriz de caracteristicas de la banda delta
filas_matriz_caracteristicas_delta.sort(key=lambda x: x[0])
matriz_caracteristicas_delta = np.array(filas_matriz_caracteristicas_delta)
matriz_caracteristicas_delta_df = pd.DataFrame(matriz_caracteristicas_delta, columns=lista_caracteristicas)
matriz_caracteristicas_delta_df.to_csv("matriz_caracteristicas_delta.csv", index=False)

# Matriz de caracteristicas de la banda theta
filas_matriz_caracteristicas_theta.sort(key=lambda x: x[0])
matriz_caracteristicas_theta = np.array(filas_matriz_caracteristicas_theta)
matriz_caracteristicas_theta_df = pd.DataFrame(matriz_caracteristicas_theta, columns=lista_caracteristicas)
matriz_caracteristicas_theta_df.to_csv("matriz_caracteristicas_theta.csv", index=False)

# Matriz de caracteristicas de la banda alpha
filas_matriz_caracteristicas_alpha.sort(key=lambda x: x[0])
matriz_caracteristicas_alpha = np.array(filas_matriz_caracteristicas_alpha)
matriz_caracteristicas_alpha_df = pd.DataFrame(matriz_caracteristicas_alpha, columns=lista_caracteristicas)
matriz_caracteristicas_alpha_df.to_csv("matriz_caracteristicas_alpha.csv", index=False)

# Matriz de caracteristicas de la banda beta
filas_matriz_caracteristicas_beta.sort(key=lambda x: x[0])
matriz_caracteristicas_beta = np.array(filas_matriz_caracteristicas_beta)
matriz_caracteristicas_beta_df = pd.DataFrame(matriz_caracteristicas_beta, columns=lista_caracteristicas)
matriz_caracteristicas_beta_df.to_csv("matriz_caracteristicas_beta.csv", index=False)

# Matriz de caracteristicas de la banda gamma
filas_matriz_caracteristicas_gamma.sort(key=lambda x: x[0])
matriz_caracteristicas_gamma = np.array(filas_matriz_caracteristicas_gamma)
matriz_caracteristicas_gamma_df = pd.DataFrame(matriz_caracteristicas_gamma, columns=lista_caracteristicas)
matriz_caracteristicas_gamma_df.to_csv("matriz_caracteristicas_gamma.csv", index=False)

# Matriz de caracteristicas de la banda full
filas_matriz_caracteristicas_full.sort(key=lambda x: x[0])
matriz_caracteristicas_full = np.array(filas_matriz_caracteristicas_full)
matriz_caracteristicas_full_df = pd.DataFrame(matriz_caracteristicas_full, columns=lista_caracteristicas)
matriz_caracteristicas_full_df.to_csv("matriz_caracteristicas_full.csv", index=False)
