import os
import mne
from mne.preprocessing import annotate_muscle_zscore, compute_proj_eog

## Carga de Datos (.fif)
ruta_archivos_cortados = os.path.join(os.getcwd(), "EEGs", "EEG_crop", "PRE")

# Recorrer todos los elementos dentro de la carpeta principal
for colegio in os.listdir(ruta_archivos_cortados):      # Recorro carpetas de colegios
    ruta_colegio = os.path.join(ruta_archivos_cortados, colegio)
    for curso in os.listdir(ruta_colegio):              # Recorro carpetas de cursos
        ruta_curso = os.path.join(ruta_colegio, curso, "EEG")
        for iniciales in os.listdir(ruta_curso):        # Recorro carpetas de iniciales
            ruta_iniciales = os.path.join(ruta_curso, iniciales)

            Session = os.listdir(ruta_iniciales)[0]
            ruta_sesion = os.path.join(ruta_iniciales, Session)

            Recording = os.listdir(ruta_sesion)[0]
            ruta_archivo = os.path.join(ruta_sesion, Recording, "aacc_vb.fif")  # Ruta al archivo FIF
            
            raw = mne.io.read_raw_fif(ruta_archivo, preload=True)  # Cargar archivo .fif (Raw data)
            
            ## Filtrado
            raw_filtered = raw.copy().pick(picks=range(32))
            raw_filtered.notch_filter(freqs=[50, 100], method='fir',fir_design='firwin')
            raw_filtered.filter(l_freq=1.0, h_freq=40.0, method='fir',fir_design='firwin')


            ## Separación en bandas canónicas
            bands = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (13, 30), "gamma": (30, 40), "full": (1, 40)}

            band_signals = {}

            for band_name, (l_freq, h_freq) in bands.items():
                band_raw = raw_filtered.copy().filter(l_freq=l_freq, h_freq=h_freq, method='fir', fir_design='firwin')
                band_signals[band_name] = band_raw


            ## Detección  y eliminación de artefactos musculares
            band_scores_muscle = {}
            threshold_muscle = 5  # Parametro ajustable

            for band_name, band_raw in band_signals.items():
                annot_muscle, scores_muscle = annotate_muscle_zscore(
                    band_raw,
                    ch_type="eeg",
                    threshold=threshold_muscle,     
                    min_length_good=0.2,
                    filter_freq=[110, 127]
                )
   
                band_raw.set_annotations(annot_muscle)  # Añadir anotaciones al objeto
                band_scores_muscle[band_name] = scores_muscle
                
            # Exclusión de artefactos musculares (Epochs)
            epochs_dict = {}

            for band_name, band_raw in band_signals.items():
                epochs = mne.make_fixed_length_epochs(
                    band_raw,
                    duration=1.0,
                    preload=True,
                    reject_by_annotation=True
                )
                epochs_dict[band_name] = epochs


            ## Eliminación de artefactos oculares
            if 'Fp1' not in raw.info['bads']:
                for band_name, band_raw in band_signals.items():
                    projs, _ = compute_proj_eog(
                        band_raw,
                        ch_name='Fp1',  # Canal usado para detectar los parpadeos
                        n_grad=0,
                        n_mag=0,
                        n_eeg=1,        # Número de componentes SSP
                        average=True
                    )

                    band_raw.add_proj(projs)    # Añadir proyecciones al objeto
                    band_raw.apply_proj()       # Aplicar SSP (atenuación de artefactos oculares)


            ## Reparación de canales malos mediante interpolación esférica
            for band_raw in band_signals.values():  
                band_raw.interpolate_bads(reset_bads=True, mode='accurate')


            ## Re-referenciación al promedio de canales (CAR)
            for band_raw in band_signals.values():
                band_raw.set_eeg_reference('average', projection=False)


            ## Guardar resultado
            ruta_archivo_preprocesado = os.path.join(os.getcwd(), "EEGs", "EEG_preprocesado", colegio, curso, "EEG", iniciales, Session, Recording)

            os.makedirs(ruta_archivo_preprocesado, exist_ok=True)
            for band_name, band_raw in band_signals.items():
                ruta_archivo_preprocesado = os.path.join(ruta_archivo_preprocesado, f"banda_{band_name}.fif")  
                band_raw.save(ruta_archivo_preprocesado, overwrite=True)
                ruta_archivo_preprocesado = ruta_archivo_preprocesado.removesuffix(f"banda_{band_name}.fif")
