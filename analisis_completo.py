import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

def calcular_distancia(lat1, lon1, lat2, lon2):
    """Calcula distancia en km usando Haversine"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return c * 6371

# Cargar CSV
print("="*70)
print("📊 ANÁLISIS COMPLETO DEL CSV DE ACCIDENTES")
print("="*70)

df = pd.read_csv('accidentes_transito_ecuador_10000.csv')

print(f"\n✅ CSV cargado exitosamente!")
print(f"📁 Total de registros: {len(df):,}")
print(f"📋 Columnas: {len(df.columns)}")

# Análisis por provincia
print("\n" + "="*70)
print("🗺️  DISTRIBUCIÓN POR PROVINCIA")
print("="*70)
for provincia, count in df['provincia'].value_counts().items():
    porcentaje = (count / len(df)) * 100
    print(f"  {provincia:20} {count:5,} ({porcentaje:5.2f}%)")

# Análisis de Azuay por ciudad
print("\n" + "="*70)
print("🏙️  DISTRIBUCIÓN EN AZUAY POR CIUDAD")
print("="*70)
df_azuay = df[df['provincia'] == 'Azuay']
print(f"Total en Azuay: {len(df_azuay):,}")
for ciudad, count in df_azuay['ciudad'].value_counts().items():
    porcentaje = (count / len(df_azuay)) * 100
    print(f"  {ciudad:20} {count:5,} ({porcentaje:5.2f}%)")

# Análisis de Cuenca específicamente
df_cuenca = df[df['ciudad'] == 'Cuenca']
print(f"\n📍 Total en Cuenca: {len(df_cuenca):,}")

# Punto de referencia (centro aproximado de Cuenca)
LAT_REF = -2.89264
LON_REF = -78.77814

# Calcular distancias para Cuenca
print("\n" + "="*70)
print(f"📏 ANÁLISIS DE DISTANCIAS DESDE PUNTO DE REFERENCIA")
print(f"   Punto: ({LAT_REF}, {LON_REF})")
print("="*70)

df_cuenca['distancia_km'] = df_cuenca.apply(
    lambda row: calcular_distancia(LAT_REF, LON_REF, row['latitud'], row['longitud']),
    axis=1
)

# Estadísticas de distancia
print(f"\n📊 Estadísticas de distancia en Cuenca:")
print(f"  Mínima:    {df_cuenca['distancia_km'].min():.2f} km")
print(f"  Máxima:    {df_cuenca['distancia_km'].max():.2f} km")
print(f"  Promedio:  {df_cuenca['distancia_km'].mean():.2f} km")
print(f"  Mediana:   {df_cuenca['distancia_km'].median():.2f} km")

# Análisis por radios
radios = [1, 2, 5, 10, 15, 20, 30, 50]

print("\n" + "="*70)
print("🎯 ACCIDENTES POR RADIO DE BÚSQUEDA (desde punto de referencia)")
print("="*70)
print(f"{'Radio (km)':>12} {'Cuenca':>12} {'Azuay Total':>15} {'Todo Ecuador':>15}")
print("-"*70)

for radio in radios:
    # Calcular para todo el dataset
    df['distancia_km'] = df.apply(
        lambda row: calcular_distancia(LAT_REF, LON_REF, row['latitud'], row['longitud']),
        axis=1
    )
    
    cuenca_en_radio = len(df_cuenca[df_cuenca['distancia_km'] <= radio])
    azuay_en_radio = len(df_azuay[df['distancia_km'] <= radio])
    total_en_radio = len(df[df['distancia_km'] <= radio])
    
    print(f"{radio:>12} {cuenca_en_radio:>12,} {azuay_en_radio:>15,} {total_en_radio:>15,}")

# Tipos de accidente
print("\n" + "="*70)
print("🚗 TIPOS DE ACCIDENTES")
print("="*70)
for tipo, count in df['tipo_accidente'].value_counts().items():
    porcentaje = (count / len(df)) * 100
    print(f"  {tipo:30} {count:5,} ({porcentaje:5.2f}%)")

# Estados
print("\n" + "="*70)
print("📋 ESTADOS DE REPORTES")
print("="*70)
for estado, count in df['estado'].value_counts().items():
    porcentaje = (count / len(df)) * 100
    print(f"  {estado:30} {count:5,} ({porcentaje:5.2f}%)")

# Víctimas y heridos
print("\n" + "="*70)
print("👥 ESTADÍSTICAS DE VÍCTIMAS")
print("="*70)
print(f"  Total víctimas mortales:  {df['victimas'].sum():,}")
print(f"  Total heridos:            {df['heridos'].sum():,}")
print(f"  Promedio víctimas/accidente: {df['victimas'].mean():.2f}")
print(f"  Promedio heridos/accidente:  {df['heridos'].mean():.2f}")

# Condiciones
print("\n" + "="*70)
print("🌤️  CONDICIONES CLIMÁTICAS")
print("="*70)
for clima, count in df['condicion_clima'].value_counts().items():
    porcentaje = (count / len(df)) * 100
    print(f"  {clima:20} {count:5,} ({porcentaje:5.2f}%)")

print("\n" + "="*70)
print("🛣️  CONDICIONES DE LA VÍA")
print("="*70)
for via, count in df['condicion_via'].value_counts().items():
    porcentaje = (count / len(df)) * 100
    print(f"  {via:20} {count:5,} ({porcentaje:5.2f}%)")

# Rango de fechas
print("\n" + "="*70)
print("📅 RANGO TEMPORAL")
print("="*70)
print(f"  Desde: {df['fecha'].min()}")
print(f"  Hasta: {df['fecha'].max()}")

# Recomendaciones
print("\n" + "="*70)
print("💡 CONCLUSIONES")
print("="*70)
print(f"  • El punto de referencia está en Cuenca")
print(f"  • Solo {len(df_cuenca[df_cuenca['distancia_km'] <= 10]):,} accidentes de Cuenca")
print(f"    están dentro de 10 km del punto de referencia")
print(f"  • Para ver MÁS accidentes, aumenta el radio_km a 20-50 km")
print(f"  • O cambia el punto de referencia a otras coordenadas")
print(f"  • El dataset tiene {len(df):,} registros totales distribuidos")
print(f"    en toda la provincia de Azuay y otras provincias")

print("\n" + "="*70)
print("✅ Análisis completado!")
print("="*70 + "\n")