import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay


## Carga de datos
banda = "alpha"         # Seleccionar banda (delta, theta, alpha, beta, gamma o full)
matriz_carac = pd.read_csv(f"matriz_caracteristicas_{banda}.csv")
meta = pd.read_csv("meta.csv")
df = matriz_carac.merge(meta[["ID", "y"]], on="ID", how="inner")
df_X = df.drop(columns=["ID", "y"])
df_y = df["y"]
y = df_y.values     # Vector de etiquetas (AACC-> y = 1, control-> y = 0)


## Definición de grupos de métricas
grupos_metricas = {
    "Descriptivas": ["media", "desviacion_estandar", "varianza", "rango", "maximo", "minimo"],
    "Diferencial": ["media_d1", "desviacion_estandar_d1", "media_d2", "desviacion_estandar_d2"],
    "Geometricas": ["valor_cuadratico_medio", "integral"],
    "Distribucion": ["asimetria", "curtosis"],
    "Espectrales": ["potencia_absoluta", "potencia_relativa_intra-canal", "potencia_relativa"],
    "Entropia": ["entropia"]
}

## Grupos de métricas para probar combinaciones
metricas_a_usar = {
     "Descriptivas": grupos_metricas["Descriptivas"],
     "Diferencial": grupos_metricas["Diferencial"],
     "Geometricas": grupos_metricas["Geometricas"],
     "Distribucion": grupos_metricas["Distribucion"],
     "Espectrales": grupos_metricas["Espectrales"],
     "Entropia": grupos_metricas["Entropia"],
    
     "Descriptivas_Diferencial": grupos_metricas["Descriptivas"] + grupos_metricas["Diferencial"],
     "Descriptivas_Geometricas": grupos_metricas["Descriptivas"] + grupos_metricas["Geometricas"],
     "Descriptivas_Distribucion": grupos_metricas["Descriptivas"] + grupos_metricas["Distribucion"],
     "Descriptivas_Espectrales": grupos_metricas["Descriptivas"] + grupos_metricas["Espectrales"],
     "Descriptivas_Entropia": grupos_metricas["Descriptivas"] + grupos_metricas["Entropia"],
     "Diferencial_Geometricas": grupos_metricas["Diferencial"] + grupos_metricas["Geometricas"],
     "Diferencial_Distribucion": grupos_metricas["Diferencial"] + grupos_metricas["Distribucion"],
     "Diferencial_Espectrales": grupos_metricas["Diferencial"] + grupos_metricas["Espectrales"],
     "Diferencial_Entropia": grupos_metricas["Diferencial"] + grupos_metricas["Entropia"],
     "Geometricas_Distribucion": grupos_metricas["Geometricas"] + grupos_metricas["Distribucion"],
     "Geometricas_Espectrales": grupos_metricas["Geometricas"] + grupos_metricas["Espectrales"],
     "Geometricas_Entropia": grupos_metricas["Geometricas"] + grupos_metricas["Entropia"],
     "Distribucion_Espectrales": grupos_metricas["Distribucion"] + grupos_metricas["Espectrales"],
     "Distribucion_Entropia": grupos_metricas["Distribucion"] + grupos_metricas["Entropia"],
     "Espectrales_Entropia": grupos_metricas["Espectrales"] + grupos_metricas["Entropia"],

     "Todas_menos_Descriptivas": grupos_metricas["Diferencial"] + grupos_metricas["Geometricas"] + grupos_metricas["Distribucion"] + grupos_metricas["Espectrales"] + grupos_metricas["Entropia"],
     "Todas_menos_Diferencial": grupos_metricas["Descriptivas"] + grupos_metricas["Geometricas"] + grupos_metricas["Distribucion"] + grupos_metricas["Espectrales"] + grupos_metricas["Entropia"],
     "Todas_menos_Geometricas": grupos_metricas["Descriptivas"] + grupos_metricas["Diferencial"] + grupos_metricas["Distribucion"] + grupos_metricas["Espectrales"] + grupos_metricas["Entropia"],
     "Todas_menos_Distribucion": grupos_metricas["Descriptivas"] + grupos_metricas["Diferencial"] + grupos_metricas["Geometricas"] + grupos_metricas["Espectrales"] + grupos_metricas["Entropia"],
     "Todas_menos_Espectrales": grupos_metricas["Descriptivas"] + grupos_metricas["Diferencial"] + grupos_metricas["Geometricas"] + grupos_metricas["Distribucion"] + grupos_metricas["Entropia"],
     "Todas_menos_Entropia": grupos_metricas["Descriptivas"] + grupos_metricas["Diferencial"] + grupos_metricas["Geometricas"] + grupos_metricas["Distribucion"] + grupos_metricas["Espectrales"],
    
     "Todas": grupos_metricas["Descriptivas"] + grupos_metricas["Diferencial"] + grupos_metricas["Geometricas"] + grupos_metricas["Distribucion"] + grupos_metricas["Espectrales"] + grupos_metricas["Entropia"]
}


## Definición de zonas cerebrales para probar combinaciones
all_channels = sorted(set(col.split('_')[0] for col in df_X.columns if '_' in col))
print(f"Canales disponibles ({len(all_channels)}): {all_channels}")
zonas = {
     "Frontal": ["Fpz", "Fz", "Fp1", "Fp2", "AF3", "AF4", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6"],
     "Frontal_Izquierda": ["Fp1", "AF3", "F3", "F7", "FC1", "FC5"],
     "Frontal_Derecha": ["Fp2", "AF4", "F4", "F8", "FC2", "FC6"],  
     "Central_Frontal": ["FC1", "FC2","FC5", "FC6"],
     "Central": ["Cz", "C3", "C4"],
     "Parietal": ["Pz", "P3", "P4", "P7", "P8", "CP1", "CP2", "CP5", "CP6", "POz"],
     "Parietal_Izquierda": ["P3", "P7", "CP1", "CP5"],
     "Parietal_Derecha": ["P4", "P8", "CP2", "CP6"],
     "Central_Parietal": ["CP1", "CP2", "CP5", "CP6"],
     "Occipital": ["POz", "Oz", "O1", "O2"],
     "Temporal": ["T7", "T8"],
     "Temporal_Izquierda": ["T7"],
     "Temporal_Derecha": ["T8"],
     "Hemisferio_Izquierdo": ["Fp1", "AF3", "F3", "F7", "FC1", "FC5", "C3", "T7", "CP1", "CP5", "P3", "P7", "O1"],
     "Hemisferio_Derecho": ["Fp2", "AF4", "F4", "F8", "FC2", "FC6", "C4", "T8", "CP2", "CP6", "P4", "P8", "O2"],
     "Todas": all_channels
}


## Bucle principal: combinaciones Zona cerebral x Grupo de métricas
resultados = []
for zona_nombre, canales_zona in zonas.items():
    for grupo_nombre, metricas_lista in metricas_a_usar.items():
        print("\n" + "="*70)
        print(f"PROCESANDO: Zona = {zona_nombre} | Grupo = {grupo_nombre}")
        print(f"Canales de la zona: {canales_zona}")
        print(f"Métricas del grupo: {metricas_lista}")
        print("="*70)


        ## Filtrar columnas según zona y grupo de métricas y obtener matriz de características reducida
        columnas_zona = [col for col in df_X.columns if col.split('_')[0] in canales_zona]
        columnas_filtradas = [col for col in columnas_zona if col.split('_', 1)[1] in metricas_lista]
        print(f"Nº de columnas resultantes: {len(columnas_filtradas)} (={len(metricas_lista)} métricas x {len(canales_zona)} canales)")

        X = df_X[columnas_filtradas].values     # Matriz de caracteristicas del subconjunto grupo de métricas / zona cerebral
        print(f"  Dimensiones X: {X.shape} (muestras x características)")


        ## Definición del pipeline
        pipeline = Pipeline([
            ('undersample', RandomUnderSampler(sampling_strategy='majority', random_state=42)), # Submuestreo
            ('scaler', StandardScaler()),                                                       # Estandarización
            ('select', SelectKBest(score_func=f_classif, k=10)),                                # Selección de carcterísticas
            ('gb', GradientBoostingClassifier(random_state=42))                                 # Modelo de clasificación
        ])


        ## Combinaciones de parámetros a probar
        param_grid = {
            'gb__learning_rate': [0.01, 0.05, 0.1],
            'gb__max_depth': [3, 5, 7],
            'gb__n_estimators': [100, 200]
        }
        

        ## Evaluación con RepeatedKFold
        num_repeats = 10
        outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=num_repeats, random_state=42)

        # Listas para almacenar métricas de cada fold
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        f1_macro_scores = []
        aucs = []
        best_params_folds = []

        # Variables para representar la matriz de confusión y curva ROC-AUC del primer Fold
        first_fold_cm = None
        first_fold_y_test = None
        first_fold_y_pred = None
        first_fold_y_prob = None
        first_fold_auc = None

        print(f"Iniciando validación cruzada ({5*num_repeats})...")
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1): 
            print(f"Fold {fold}/{5*num_repeats}")

            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            # Búsqueda de hiperparámetros con GridSearchCV
            cv_grid = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv_grid,
                scoring='f1_macro',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            best_params_fold = grid_search.best_params_
            best_params_folds.append(best_params_fold)

            best_f1_macro_inner = grid_search.best_score_
            print(" Mejores parámetros:", best_params_fold)
            print(f"  Mejor F1 macro (validación cruzada interna): {best_f1_macro_inner:.3f}")

            # Definición y entrenamiento del modelo
            model = clone(pipeline)
            model.set_params(**best_params_fold) # Modelo con los hiperparámetros seleccionados con GridSearch
            model.fit(X_train, y_train)

            # Prueba del modelo
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Métricas del fold
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            auc = roc_auc_score(y_test, y_prob)

            # Almacenar
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            f1_macro_scores.append(f1_macro)
            aucs.append(auc)

            # Guardo valores del primer fold para representación gráfica
            if fold == 1:
                first_fold_cm = confusion_matrix(y_test, y_pred)
                first_fold_y_test = y_test
                first_fold_y_pred = y_pred
                first_fold_y_prob = y_prob
                first_fold_auc = auc

            print(
                f"ACC={accuracy:.3f} | "
                f"F1={f1:.3f} | "
                f"F1_macro={f1_macro:.3f} | "
                f"AUC={auc:.3f}"
            )


        ## Obtención de parámetros más frecuentes
        params_as_tuples = [
            tuple(sorted(p.items()))
            for p in best_params_folds
        ]

        contador = Counter(params_as_tuples)

        most_common_params_tuple, frecuencia = contador.most_common(1)[0]

        most_common_params = dict(most_common_params_tuple)

        print("\nParámetros más frecuentes:")
        print(most_common_params)
        print(f"Seleccionados en {frecuencia}/{5*num_repeats} folds")


        ## Almacenar resultados de esta combinación Features/Zona
        resultados.append({
            "Features": grupo_nombre,
            "Zona": zona_nombre,
            "Num_caracteristicas": X.shape[1],
            "Accuracy_mean": np.mean(accuracies),
            "Accuracy_std": np.std(accuracies),
            "F1_mean": np.mean(f1_scores),
            "F1_std": np.std(f1_scores),
            "F1_macro_mean": np.mean(f1_macro_scores),
            "F1_macro_std": np.std(f1_macro_scores),
            "Precision_mean": np.mean(precisions),
            "Precision_std": np.std(precisions),
            "Recall_mean": np.mean(recalls),
            "Recall_std": np.std(recalls),
            "AUC_mean": np.mean(aucs),
            "AUC_std": np.std(aucs),
            "Mejores_parametros": str(most_common_params),
            "Frecuencia_mejores_parametros": f"{frecuencia}/{5*num_repeats}"
        })

        
        ## Representaciones gráficas para el primer fold de esta combinación
        if first_fold_cm is not None:
            # Matriz de confusión
            fig, ax = plt.subplots(1,2, figsize=(12,5))
            ConfusionMatrixDisplay(first_fold_cm, display_labels=['Control (0)','AACC (1)']).plot(ax=ax[0], cmap='Blues', values_format='d')
            ax[0].set_title(f"Confusion Matrix - {banda}")
            
            ConfusionMatrixDisplay.from_predictions(first_fold_y_test, first_fold_y_pred, display_labels=['Control (0)','AACC (1)'], normalize='true', ax=ax[1], cmap='Blues')
            ax[1].set_title(f"Normalized Confusion Matrix - {banda}")
            plt.tight_layout()
            plt.savefig(f"matriz_confusion_{banda}_{zona_nombre}_{grupo_nombre}.png", dpi=300)
            plt.close()
            
            # Curva ROC
            fpr, tpr, _ = roc_curve(first_fold_y_test, first_fold_y_prob)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC = {first_fold_auc:.3f}")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {banda}")
            plt.legend()
            plt.savefig(f"curva_roc_{banda}_{zona_nombre}_{grupo_nombre}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"  Resultados para {zona_nombre} / {grupo_nombre}: F1 = {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        

## Tabla comparativa y exportación a CSV
print("\n" + "="*70)
print("RESUMEN COMPARATIVO POR FEATURES / ZONA")
print("="*70)

df_resultados = pd.DataFrame(resultados)
# Crear columnas de texto con formato "media ± std"
df_resultados["Accuracy"] = df_resultados.apply(lambda r: f"{r['Accuracy_mean']:.4f} ± {r['Accuracy_std']:.4f}", axis=1)
df_resultados["F1-score"] = df_resultados.apply(lambda r: f"{r['F1_mean']:.4f} ± {r['F1_std']:.4f}", axis=1)
df_resultados["F1-macro"] = df_resultados.apply(lambda r: f"{r['F1_macro_mean']:.4f} ± {r['F1_macro_std']:.4f}", axis=1)
df_resultados["Precision"] = df_resultados.apply(lambda r: f"{r['Precision_mean']:.4f} ± {r['Precision_std']:.4f}", axis=1)
df_resultados["Recall"] = df_resultados.apply(lambda r: f"{r['Recall_mean']:.4f} ± {r['Recall_std']:.4f}", axis=1)
df_resultados["AUC"] = df_resultados.apply(lambda r: f"{r['AUC_mean']:.4f} ± {r['AUC_std']:.4f}", axis=1)

# Seleccionar columnas para mostrar
columnas_mostrar = ["Features", "Zona", "Num_caracteristicas", "Accuracy", "F1-score", "F1-macro", "Precision", "Recall", "AUC", "Mejores_parametros", "Frecuencia_mejores_parametros"]
df_mostrar = df_resultados[columnas_mostrar].copy()
# Ordenar por F1 medio descendente
df_mostrar["F1_orden"] = df_resultados["F1_macro_mean"]
df_mostrar = df_mostrar.sort_values("F1_orden", ascending=False).drop(columns="F1_orden")

print(df_mostrar.to_string(index=False))

# Exportar a CSV
nombre_csv = f"resultados_{banda}.csv"
df_mostrar.to_csv(nombre_csv, index=False, encoding="utf-8")
print(f"\nResultados guardados en '{nombre_csv}'")


# Exportar a Excel
nombre_excel = f"resultados_{banda}.xlsx"

with pd.ExcelWriter(nombre_excel, engine="openpyxl") as writer:
    df_mostrar.to_excel(writer, sheet_name="Resultados", index=False)

    ws = writer.sheets["Resultados"]

    # Autofiltro
    ws.auto_filter.ref = ws.dimensions

    # Ajustar ancho de columnas
    for column in ws.columns:
        max_length = 0
        column_letter = column[0].column_letter

        for cell in column:
            try:
                max_length = max(max_length, len(str(cell.value)))
            except:
                pass

        ws.column_dimensions[column_letter].width = max_length + 2

print(f"Resultados guardados en '{nombre_excel}'")

