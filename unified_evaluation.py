#!/usr/bin/env python3
"""
Script unificado para evaluar y comparar todos los modelos con el mismo split
"""

import argparse
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras

# Configuración del grid físico
GRID_SHAPE = (6, 4)
INDEX_MAP_ROWS = [
    [1, 2, 7, 8],
    [3, 4, 9, 10],
    [5, 6, 11, 12],
    [13, 14, 19, 20],
    [15, 16, 21, 22],
    [17, 18, 23, 24],
]
PERM_24 = [idx - 1 for row in INDEX_MAP_ROWS for idx in row]

def load_unified_split(split_path):
    """Cargar el split unificado"""
    with open(split_path, 'rb') as f:
        split_data = pickle.load(f)
    
    return (split_data['X_train'], split_data['X_test'], 
            split_data['y_train'], split_data['y_test'])

def evaluate_mlp_baseline(X_test, y_test, model_path, scalers_path):
    """Evaluar MLP Baseline"""
    print("🧠 Evaluando MLP Baseline...")
    
    # Cargar modelo y scalers
    model = keras.models.load_model(model_path, compile=False)
    with open(scalers_path, 'rb') as f:
        scaler_X, scaler_y = pickle.load(f)
    
    # Transformar datos
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Predicciones
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test_scaled)
    
    # Métricas
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"   MLP Baseline -> MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    return {
        'model': 'MLP Baseline',
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'y_true': y_true,
        'y_pred': y_pred
    }

def evaluate_cnn2d(X_test, y_test, model_path, scalers_path):
    """Evaluar CNN 2D"""
    print("🔍 Evaluando CNN 2D...")
    
    # Cargar modelo y scalers
    model = keras.models.load_model(model_path, compile=False)
    with open(scalers_path, 'rb') as f:
        scaler_X, scaler_y = pickle.load(f)
    
    # Escalar en formato plano (24) y rearmar a imagen para la entrada
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, 24)).reshape(-1, *GRID_SHAPE, 1)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 24))
    
    # Predicciones (salida plana de 24)
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test_scaled)
    
    # Métricas
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"   CNN 2D -> MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    return {
        'model': 'CNN 2D',
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'y_true': y_true,
        'y_pred': y_pred
    }

def evaluate_cnn2d_attention(X_test, y_test, model_path, scalers_path):
    """Evaluar CNN 2D + Attention"""
    print("🎯 Evaluando CNN 2D + Attention...")
    
    # Reshape entrada a imagen, salida plana
    X_test_cnn = X_test.reshape(-1, *GRID_SHAPE, 1)
    
    model = keras.models.load_model(model_path, compile=False)
    with open(scalers_path, 'rb') as f:
        scaler_X, scaler_y = pickle.load(f)
    
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, 24)).reshape(-1, *GRID_SHAPE, 1)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 24))
    
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test_scaled)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"   CNN 2D + Attention -> MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    return {
        'model': 'CNN 2D + Attention',
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'y_true': y_true,
        'y_pred': y_pred
    }

def create_comparison_table(results, output_dir):
    """Crear tabla comparativa"""
    print("\n📋 Creando tabla comparativa...")
    
    data = []
    for result in results:
        data.append({
            'Modelo': result['model'],
            'MSE': f"{result['mse']:.4f}",
            'RMSE': f"{result['rmse']:.4f}",
            'R²': f"{result['r2']:.4f}",
            'MAE': f"{result['mae']:.4f}"
        })
    
    df = pd.DataFrame(data)
    
    # Guardar CSV
    csv_path = os.path.join(output_dir, 'comparison_table.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Tabla guardada en: {csv_path}")
    
    # Mostrar tabla
    print("\n=== TABLA COMPARATIVA ===")
    print(df.to_string(index=False))
    
    return df

def create_comparison_table_by_distance(all_results, output_dir):
    """Crear tabla comparativa multi-distancia.
    all_results: lista de dicts con claves {'distance_cm', 'model', 'mse','rmse','r2','mae'}
    """
    print("\n📋 Creando tabla comparativa por distancia...")
    rows = []
    for r in all_results:
        rows.append({
            'Distance_cm': r['distance_cm'],
            'Model': r['model'],
            'MSE': f"{r['mse']:.4f}",
            'RMSE': f"{r['rmse']:.4f}",
            'R²': f"{r['r2']:.4f}",
            'MAE': f"{r['mae']:.4f}",
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(by=['Distance_cm','Model'])
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'comparison_by_distance.csv')
    df.to_csv(csv_path, index=False)
    print(f"Tabla multi-distancia guardada en: {csv_path}")
    return df

def plot_comparison(results, output_dir):
    """Crear gráficas de comparación"""
    print("\n📊 Generando gráficas...")
    
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Gráfica de métricas
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    models = [r['model'] for r in results]
    mse_values = [r['mse'] for r in results]
    rmse_values = [r['rmse'] for r in results]
    r2_values = [r['r2'] for r in results]
    mae_values = [r['mae'] for r in results]
    
    # MSE
    axes[0,0].bar(models, mse_values, color='skyblue', alpha=0.7)
    axes[0,0].set_title('MSE (Mean Squared Error)')
    axes[0,0].set_ylabel('MSE')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # RMSE
    axes[0,1].bar(models, rmse_values, color='lightcoral', alpha=0.7)
    axes[0,1].set_title('RMSE (Root Mean Squared Error)')
    axes[0,1].set_ylabel('RMSE')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # R²
    axes[1,0].bar(models, r2_values, color='lightgreen', alpha=0.7)
    axes[1,0].set_title('R² Score')
    axes[1,0].set_ylabel('R²')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # MAE
    axes[1,1].bar(models, mae_values, color='gold', alpha=0.7)
    axes[1,1].set_title('MAE (Mean Absolute Error)')
    axes[1,1].set_ylabel('MAE')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 2. Gráfica de predicciones vs reales
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for i, result in enumerate(results):
        y_true = result['y_true'].flatten()
        y_pred = result['y_pred'].flatten()
        
        axes[i].scatter(y_true, y_pred, alpha=0.5, s=1)
        
        # Línea perfecta
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        axes[i].plot(lims, lims, 'r--', linewidth=2)
        
        axes[i].set_xlabel('True Values')
        axes[i].set_ylabel('Predicted Values')
        axes[i].set_title(f"{result['model']}\nR² = {result['r2']:.4f}")
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions_vs_true.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    print("✅ Gráficas guardadas")

def plot_sample_heatmaps(results, output_dir, num_samples: int = 6):
    """Genera heatmaps 6x4 por muestra: primera columna es 'Real' y
    las siguientes columnas son las predicciones de cada modelo.
    Guarda una figura por muestra como PNG.
    """
    if not results:
        return
    os.makedirs(output_dir, exist_ok=True)
    
    # Tomamos y_true desde el primer resultado (todas deben tener el mismo orden)
    y_true_all = results[0]['y_true']  # shape: (N, 24)
    N = y_true_all.shape[0]
    idxs = np.linspace(0, N - 1, num=min(num_samples, N), dtype=int)
    
    num_models = len(results)
    cols = 1 + num_models  # Real + cada modelo
    
    for k, idx in enumerate(idxs):
        fig, axes = plt.subplots(1, cols, figsize=(4.2 * cols, 4.0))
        if cols == 1:
            axes = [axes]
        
        # Panel 0: Real
        true_grid = y_true_all[idx].reshape(GRID_SHAPE[0], GRID_SHAPE[1])
        im0 = axes[0].imshow(true_grid, cmap='viridis', interpolation='nearest', aspect='equal')
        axes[0].set_title('Real (m/s)')
        axes[0].set_xticks([]); axes[0].set_yticks([])
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Siguientes paneles: predicciones por modelo
        for j, res in enumerate(results, start=1):
            y_pred_all = res['y_pred']  # (N, 24)
            pred_grid = y_pred_all[idx].reshape(GRID_SHAPE[0], GRID_SHAPE[1])
            im = axes[j].imshow(pred_grid, cmap='viridis', interpolation='nearest', aspect='equal')
            axes[j].set_title(f"{res['model']}")
            axes[j].set_xticks([]); axes[j].set_yticks([])
            plt.colorbar(im, ax=axes[j], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        out_path = os.path.join(output_dir, f'sample_heatmaps_{k}_idx_{idx}.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

def statistical_analysis(results):
    """Análisis estadístico"""
    print("\n📈 Análisis estadístico...")
    
    # Encontrar mejor modelo
    best_model = max(results, key=lambda x: x['r2'])
    
    print(f"\n🏆 MEJOR MODELO: {best_model['model']}")
    print(f"   R²: {best_model['r2']:.4f}")
    print(f"   MSE: {best_model['mse']:.4f}")
    print(f"   RMSE: {best_model['rmse']:.4f}")
    print(f"   MAE: {best_model['mae']:.4f}")
    
    # Comparaciones
    print(f"\n=== COMPARACIONES ===")
    for i, model1 in enumerate(results):
        for j, model2 in enumerate(results):
            if i < j:
                r2_diff = model2['r2'] - model1['r2']
                improvement = (r2_diff / model1['r2']) * 100 if model1['r2'] != 0 else 0
                
                print(f"{model2['model']} vs {model1['model']}:")
                print(f"  Diferencia R²: {r2_diff:.4f}")
                print(f"  Mejora: {improvement:.2f}%")
                print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', type=str, default='/home/dani/Desktop/Projects/wind_tunnel/unified_models')
    parser.add_argument('--out', type=str, default='/home/dani/Desktop/Projects/wind_tunnel/unified_comparison')
    parser.add_argument('--distance', type=float, default=None, help='Evaluar una única distancia (modo simple)')
    parser.add_argument('--distances', nargs='+', type=float, default=None,
                        help='Lista de distancias a evaluar, p.ej.: 18 28 38 48 58 68 78')
    parser.add_argument('--models', nargs='+', default=['mlp', 'cnn2d'], 
                       choices=['mlp', 'cnn2d', 'attention'],
                       help='Modelos a evaluar')
    args = parser.parse_args()
    
    print("🔬 EVALUACIÓN UNIFICADA DE MODELOS")
    print("=" * 50)
    print(f"Directorio base de modelos: {args.models_dir}")
    print(f"Modelos: {args.models}")
    print(f"Salida: {args.out}")
    print()

    # Crear directorio de salida
    os.makedirs(args.out, exist_ok=True)

    # Determinar distancias a evaluar
    distances = args.distances
    if distances is None:
        distances = [args.distance] if args.distance is not None else None

    if distances is None:
        # Modo legacy: mismos archivos en carpeta base
        split_path = os.path.join(args.models_dir, 'unified_split.pkl')
        if not os.path.exists(split_path):
            print(f"❌ No se encontró el split unificado en: {split_path}")
            print("   Ejecuta primero unified_training.py")
            return
        X_train, X_test, y_train, y_test = load_unified_split(split_path)

        results = []
        if 'mlp' in args.models:
            mlp_path = os.path.join(args.models_dir, 'mlp_baseline.keras')
            mlp_scalers = os.path.join(args.models_dir, 'mlp_scalers.pkl')
            if os.path.exists(mlp_path) and os.path.exists(mlp_scalers):
                results.append(evaluate_mlp_baseline(X_test, y_test, mlp_path, mlp_scalers))
            else:
                print("⚠️  MLP Baseline no encontrado")
        if 'cnn2d' in args.models:
            cnn_path = os.path.join(args.models_dir, 'cnn2d.keras')
            cnn_scalers = os.path.join(args.models_dir, 'cnn2d_scalers.pkl')
            if os.path.exists(cnn_path) and os.path.exists(cnn_scalers):
                results.append(evaluate_cnn2d(X_test, y_test, cnn_path, cnn_scalers))
            else:
                print("⚠️  CNN 2D no encontrado")
        if 'attention' in args.models:
            att_path = os.path.join(args.models_dir, 'cnn2d_attention.keras')
            att_scalers = os.path.join(args.models_dir, 'cnn2d_attention_scalers.pkl')
            if os.path.exists(att_path) and os.path.exists(att_scalers):
                results.append(evaluate_cnn2d_attention(X_test, y_test, att_path, att_scalers))
            else:
                print("⚠️  CNN 2D + Attention no encontrado")

        if not results:
            print("❌ No se encontraron modelos para evaluar")
            return

        create_comparison_table(results, args.out)
        plot_comparison(results, args.out)
        plot_sample_heatmaps(results, args.out, num_samples=8)
        statistical_analysis(results)
        print(f"\n✅ Evaluación completada. Resultados en: {args.out}")
        return

    # Modo multi-distancia
    all_rows = []
    for dist in distances:
        dist_models_dir = os.path.join(args.models_dir, f"{int(dist)}cm")
        dist_out_dir = os.path.join(args.out, f"{int(dist)}cm")
        os.makedirs(dist_out_dir, exist_ok=True)

        print("-" * 50)
        print(f"📏 Distancia: {dist} cm")
        print(f"📂 Modelos: {dist_models_dir}")
        print(f"📤 Salida: {dist_out_dir}")

        split_path = os.path.join(dist_models_dir, 'unified_split.pkl')
        if not os.path.exists(split_path):
            print(f"⚠️  Split no encontrado para {dist} cm en {split_path}. Saltando.")
            continue
        X_train, X_test, y_train, y_test = load_unified_split(split_path)

        results = []
        if 'mlp' in args.models:
            mlp_path = os.path.join(dist_models_dir, 'mlp_baseline.keras')
            mlp_scalers = os.path.join(dist_models_dir, 'mlp_scalers.pkl')
            if os.path.exists(mlp_path) and os.path.exists(mlp_scalers):
                res = evaluate_mlp_baseline(X_test, y_test, mlp_path, mlp_scalers)
                results.append(res)
                all_rows.append({**res, 'distance_cm': dist})
            else:
                print("⚠️  MLP Baseline no encontrado")
        if 'cnn2d' in args.models:
            cnn_path = os.path.join(dist_models_dir, 'cnn2d.keras')
            cnn_scalers = os.path.join(dist_models_dir, 'cnn2d_scalers.pkl')
            if os.path.exists(cnn_path) and os.path.exists(cnn_scalers):
                res = evaluate_cnn2d(X_test, y_test, cnn_path, cnn_scalers)
                results.append(res)
                all_rows.append({**res, 'distance_cm': dist})
            else:
                print("⚠️  CNN 2D no encontrado")
        if 'attention' in args.models:
            att_path = os.path.join(dist_models_dir, 'cnn2d_attention.keras')
            att_scalers = os.path.join(dist_models_dir, 'cnn2d_attention_scalers.pkl')
            if os.path.exists(att_path) and os.path.exists(att_scalers):
                res = evaluate_cnn2d_attention(X_test, y_test, att_path, att_scalers)
                results.append(res)
                all_rows.append({**res, 'distance_cm': dist})
            else:
                print("⚠️  CNN 2D + Attention no encontrado")

        if not results:
            print(f"⚠️  Sin resultados en {dist} cm")
            continue

        # Salidas por distancia
        create_comparison_table(results, dist_out_dir)
        plot_comparison(results, dist_out_dir)
        plot_sample_heatmaps(results, dist_out_dir, num_samples=8)
        statistical_analysis(results)

    if all_rows:
        create_comparison_table_by_distance(all_rows, args.out)
        print(f"\n✅ Evaluación multi-distancia completada. Resultados en: {args.out}")
    else:
        print("❌ No se generaron resultados multi-distancia")

if __name__ == '__main__':
    main()
