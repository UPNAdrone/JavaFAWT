#!/usr/bin/env python3
"""
Script unificado para entrenar todos los modelos con el mismo split de datos
"""

import argparse
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

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

def load_and_prepare_data(csv_path: str, distance_cm: float):
    """Cargar y preparar datos con split unificado"""
    print(f"📊 Cargando datos para distancia: {distance_cm} cm")
    
    df = pd.read_csv(csv_path)
    df = df[df['Distancia (cm)'] == distance_cm].reset_index(drop=True)
    
    if len(df) == 0:
        raise ValueError(f'Sin datos para distancia {distance_cm} cm')
    
    fan_cols = [c for c in df.columns if c.startswith('Ventilador')]
    vel_cols = [c for c in df.columns if c.startswith('Velocidad')]
    
    X = df[fan_cols].values.astype(np.float32)  # Potencias (0-100%)
    Y = df[vel_cols].values.astype(np.float32)   # Velocidades (m/s)
    
    # Reordenar a layout físico
    X = X[:, PERM_24]
    Y = Y[:, PERM_24]
    
    print(f"✅ Datos cargados: {len(X)} muestras")
    print(f"   Rango de potencias: {X.min():.1f} - {X.max():.1f} %")
    print(f"   Rango de velocidades: {Y.min():.3f} - {Y.max():.3f} m/s")
    
    # Split unificado (mismo para todos los modelos)
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    print(f"   Split: {len(X_train)} entrenamiento, {len(X_test)} prueba")
    
    return X_train, X_test, y_train, y_test

def save_unified_split(X_train, X_test, y_train, y_test, output_dir):
    """Guardar el split unificado para uso posterior"""
    os.makedirs(output_dir, exist_ok=True)
    
    split_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'random_seed': RANDOM_SEED
    }
    
    split_path = os.path.join(output_dir, 'unified_split.pkl')
    with open(split_path, 'wb') as f:
        pickle.dump(split_data, f)
    
    print(f"💾 Split unificado guardado en: {split_path}")
    return split_path

def train_mlp_baseline(X_train, y_train, X_test, y_test, output_dir):
    """Entrenar MLP Baseline"""
    print("\n🧠 Entrenando MLP Baseline...")
    
    # Normalización
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Modelo MLP
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(y_train.shape[1], activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # Entrenamiento
    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_test_scaled, y_test_scaled),
        epochs=200,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluación
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test_scaled)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"   MLP Baseline -> MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    # Guardar modelo y scalers
    model_path = os.path.join(output_dir, 'mlp_baseline.keras')
    scalers_path = os.path.join(output_dir, 'mlp_scalers.pkl')
    
    model.save(model_path)
    with open(scalers_path, 'wb') as f:
        pickle.dump((scaler_X, scaler_y), f)
    
    return {
        'model_path': model_path,
        'scalers_path': scalers_path,
        'metrics': {'mse': mse, 'rmse': rmse, 'r2': r2, 'mae': mae}
    }

def train_cnn2d(X_train, y_train, X_test, y_test, output_dir):
    """Entrenar CNN 2D"""
    print("\n🔍 Entrenando CNN 2D...")
    
    # Reshape para CNN (samples, height, width, channels)
    X_train_cnn = X_train.reshape(-1, *GRID_SHAPE, 1)
    X_test_cnn = X_test.reshape(-1, *GRID_SHAPE, 1)
    
    # Normalización sobre vectores de 24 características por muestra
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_flat = X_train.reshape(-1, 24)
    X_test_flat = X_test.reshape(-1, 24)
    y_train_flat = y_train.reshape(-1, 24)
    y_test_flat = y_test.reshape(-1, 24)
    
    X_train_scaled_flat = scaler_X.fit_transform(X_train_flat)
    X_test_scaled_flat = scaler_X.transform(X_test_flat)
    y_train_scaled_flat = scaler_y.fit_transform(y_train_flat)
    y_test_scaled_flat = scaler_y.transform(y_test_flat)
    
    # Volver a imagen para entrada, mantener y en plano (24) para la pérdida
    X_train_scaled = X_train_scaled_flat.reshape(-1, *GRID_SHAPE, 1)
    X_test_scaled = X_test_scaled_flat.reshape(-1, *GRID_SHAPE, 1)
    y_train_scaled = y_train_scaled_flat  # (n, 24)
    y_test_scaled = y_test_scaled_flat    # (n, 24)
    
    # Modelo CNN 2D
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(*GRID_SHAPE, 1)),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(24, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # Entrenamiento
    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_test_scaled, y_test_scaled),
        epochs=200,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluación
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)  # (n, 24)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test_scaled)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"   CNN 2D -> MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    # Guardar modelo y scalers
    model_path = os.path.join(output_dir, 'cnn2d.keras')
    scalers_path = os.path.join(output_dir, 'cnn2d_scalers.pkl')
    
    model.save(model_path)
    with open(scalers_path, 'wb') as f:
        pickle.dump((scaler_X, scaler_y), f)
    
    return {
        'model_path': model_path,
        'scalers_path': scalers_path,
        'metrics': {'mse': mse, 'rmse': rmse, 'r2': r2, 'mae': mae}
    }

def train_cnn2d_attention(X_train, y_train, X_test, y_test, output_dir):
    """Entrenar CNN 2D + Attention (self-attn simple)"""
    print("\n🎯 Entrenando CNN 2D + Attention...")
    
    # Reshape entradas a imagen, salidas planas
    X_train_cnn = X_train.reshape(-1, *GRID_SHAPE, 1)
    X_test_cnn = X_test.reshape(-1, *GRID_SHAPE, 1)
    
    # Normalización plana (24)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_flat = X_train.reshape(-1, 24)
    X_test_flat = X_test.reshape(-1, 24)
    y_train_flat = y_train.reshape(-1, 24)
    y_test_flat = y_test.reshape(-1, 24)
    
    X_train_scaled_flat = scaler_X.fit_transform(X_train_flat)
    X_test_scaled_flat = scaler_X.transform(X_test_flat)
    y_train_scaled_flat = scaler_y.fit_transform(y_train_flat)
    y_test_scaled_flat = scaler_y.transform(y_test_flat)
    
    X_train_scaled = X_train_scaled_flat.reshape(-1, *GRID_SHAPE, 1)
    X_test_scaled = X_test_scaled_flat.reshape(-1, *GRID_SHAPE, 1)
    y_train_scaled = y_train_scaled_flat
    y_test_scaled = y_test_scaled_flat
    
    # Bloque CNN base
    inputs = keras.Input(shape=(*GRID_SHAPE, 1))
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    
    # Atención espacial tipo self-attention simple: aplanar y aplicar MultiHeadAttention
    b, h, w, c = None, GRID_SHAPE[0]//2+ (GRID_SHAPE[0]%2==0)*0, GRID_SHAPE[1]//2+ (GRID_SHAPE[1]%2==0)*0, 128
    # En práctica, tomamos la forma de x en runtime
    s = layers.Reshape((-1, 128))(x)  # (batch, tokens, channels)
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(s, s)
    s = layers.LayerNormalization()(s + attn)
    s = layers.Dense(128, activation='relu')(s)
    s = layers.GlobalAveragePooling1D()(s)
    
    x = layers.Dense(128, activation='relu')(s)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(24, activation='linear')(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
    )
    
    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_test_scaled, y_test_scaled),
        epochs=200,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test_scaled)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"   CNN 2D + Attention -> MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    model_path = os.path.join(output_dir, 'cnn2d_attention.keras')
    scalers_path = os.path.join(output_dir, 'cnn2d_attention_scalers.pkl')
    model.save(model_path)
    with open(scalers_path, 'wb') as f:
        pickle.dump((scaler_X, scaler_y), f)
    
    return {
        'model_path': model_path,
        'scalers_path': scalers_path,
        'metrics': {'mse': mse, 'rmse': rmse, 'r2': r2, 'mae': mae}
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--distance', type=float, default=78.0, help='Distancia objetivo en cm (modo simple)')
    parser.add_argument('--distances', nargs='+', type=float, default=None,
                        help='Lista de distancias a procesar en cm, p.ej.: 18 28 38 48 58 68 78')
    parser.add_argument('--csv', type=str, default='/home/dani/Desktop/Projects/wind_tunnel/DatasetV2_velocities.csv')
    parser.add_argument('--out', type=str, default='/home/dani/Desktop/Projects/wind_tunnel/unified_models')
    parser.add_argument('--models', nargs='+', default=['mlp', 'cnn2d'], 
                       choices=['mlp', 'cnn2d', 'attention'],
                       help='Modelos a entrenar')
    args = parser.parse_args()
    
    print("🚀 ENTRENAMIENTO UNIFICADO DE MODELOS")
    print("=" * 50)
    print(f"Modelos: {args.models}")
    print(f"Salida base: {args.out}")
    print()

    # Determinar distancias a procesar
    distances = args.distances if args.distances is not None else [args.distance]

    for dist in distances:
        dist_dir = os.path.join(args.out, f"{int(dist)}cm")
        os.makedirs(dist_dir, exist_ok=True)

        print("-" * 50)
        print(f"📏 Distancia: {dist} cm")
        print(f"📂 Carpeta de salida: {dist_dir}")
        
        # Cargar y preparar datos para esta distancia
        X_train, X_test, y_train, y_test = load_and_prepare_data(args.csv, dist)
        
        # Guardar split unificado específico de la distancia
        split_path = save_unified_split(X_train, X_test, y_train, y_test, dist_dir)
        
        results = {}
        
        # Entrenar modelos seleccionados
        if 'mlp' in args.models:
            results['mlp'] = train_mlp_baseline(X_train, y_train, X_test, y_test, dist_dir)
        
        if 'cnn2d' in args.models:
            results['cnn2d'] = train_cnn2d(X_train, y_train, X_test, y_test, dist_dir)
        
        if 'attention' in args.models:
            results['attention'] = train_cnn2d_attention(X_train, y_train, X_test, y_test, dist_dir)
        
        # Resumen final por distancia
        print("\n" + "=" * 50)
        print(f"📊 RESUMEN DE RESULTADOS - {dist} cm")
        print("=" * 50)
        for model_name, result in results.items():
            metrics = result['metrics']
            print(f"{model_name.upper()}:")
            print(f"  R²: {metrics['r2']:.4f}")
            print(f"  MSE: {metrics['mse']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")
            print()
        print(f"✅ Modelos guardados en: {dist_dir}")
        print(f"💾 Split unificado en: {split_path}")

if __name__ == '__main__':
    main()
