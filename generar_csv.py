import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configuración de datos
np.random.seed(42)
random.seed(42)

# Tipos de accidentes
TIPOS_ACCIDENTE = [
    "Colisión frontal",
    "Colisión lateral",
    "Atropello",
    "Volcamiento",
    "Choque con objeto fijo",
    "Caída de pasajero",
    "Salida de vía",
    "Colisión múltiple"
]

# Provincias de Ecuador con pesos (Azuay tendrá 70% de los datos)
PROVINCIAS = {
    "Azuay": 0.70,
    "Guayas": 0.10,
    "Pichincha": 0.08,
    "Manabí": 0.05,
    "El Oro": 0.04,
    "Loja": 0.03
}

# Ciudades por provincia (mayoría en Cuenca)
CIUDADES = {
    "Azuay": {
        "Cuenca": 0.85,  # 85% de los accidentes de Azuay en Cuenca
        "Gualaceo": 0.08,
        "Paute": 0.04,
        "Sigsig": 0.03
    },
    "Guayas": {
        "Guayaquil": 0.90,
        "Durán": 0.07,
        "Milagro": 0.03
    },
    "Pichincha": {
        "Quito": 0.95,
        "Sangolquí": 0.05
    },
    "Manabí": {
        "Manta": 0.60,
        "Portoviejo": 0.40
    },
    "El Oro": {
        "Machala": 0.80,
        "Huaquillas": 0.20
    },
    "Loja": {
        "Loja": 0.90,
        "Catamayo": 0.10
    }
}

# Coordenadas aproximadas de ciudades principales
COORDENADAS_CIUDADES = {
    "Cuenca": {"lat_base": -2.9001, "lon_base": -79.0059, "radio": 0.05},
    "Gualaceo": {"lat_base": -2.8977, "lon_base": -78.7767, "radio": 0.02},
    "Paute": {"lat_base": -2.7833, "lon_base": -78.7500, "radio": 0.015},
    "Sigsig": {"lat_base": -3.0500, "lon_base": -78.8000, "radio": 0.015},
    "Guayaquil": {"lat_base": -2.1894, "lon_base": -79.8883, "radio": 0.08},
    "Durán": {"lat_base": -2.1819, "lon_base": -79.8271, "radio": 0.02},
    "Milagro": {"lat_base": -2.1344, "lon_base": -79.5939, "radio": 0.02},
    "Quito": {"lat_base": -0.1807, "lon_base": -78.4678, "radio": 0.08},
    "Sangolquí": {"lat_base": -0.3075, "lon_base": -78.4486, "radio": 0.015},
    "Manta": {"lat_base": -0.9677, "lon_base": -80.7089, "radio": 0.03},
    "Portoviejo": {"lat_base": -1.0543, "lon_base": -80.4553, "radio": 0.03},
    "Machala": {"lat_base": -3.2581, "lon_base": -79.9553, "radio": 0.03},
    "Huaquillas": {"lat_base": -3.4758, "lon_base": -80.2308, "radio": 0.015},
    "Loja": {"lat_base": -3.9930, "lon_base": -79.2040, "radio": 0.03},
    "Catamayo": {"lat_base": -3.9833, "lon_base": -79.3667, "radio": 0.015}
}

ESTADOS = ["REPORTADO", "EN_INVESTIGACION", "CERRADO", "ARCHIVADO"]

def generar_coordenadas(ciudad: str) -> tuple:
    """Genera coordenadas aleatorias alrededor de una ciudad"""
    coords = COORDENADAS_CIUDADES[ciudad]
    lat = coords["lat_base"] + np.random.uniform(-coords["radio"], coords["radio"])
    lon = coords["lon_base"] + np.random.uniform(-coords["radio"], coords["radio"])
    return round(lat, 7), round(lon, 7)

def generar_direccion(ciudad: str, provincia: str) -> str:
    """Genera una dirección aproximada"""
    calles = ["Av. Principal", "Calle Central", "Av. Loja", "Calle Bolívar", 
              "Av. España", "Calle Sucre", "Av. 10 de Agosto", "Calle Ordóñez"]
    numeros = ["y Calle Solano", "y Av. González Suárez", "sector norte",
               "sector sur", "centro histórico", "parque industrial"]
    
    return f"{random.choice(calles)} {random.choice(numeros)}, {ciudad}, {provincia}, Ecuador"

def generar_dataset(n_registros: int = 10000):
    """Genera el dataset completo de accidentes"""
    
    datos = []
    fecha_inicio = datetime(2023, 1, 1)
    fecha_fin = datetime(2026, 1, 3)
    
    for i in range(1, n_registros + 1):
        # Seleccionar provincia según distribución
        provincia = random.choices(
            list(PROVINCIAS.keys()),
            weights=list(PROVINCIAS.values())
        )[0]
        
        # Seleccionar ciudad según distribución
        ciudades_prov = CIUDADES[provincia]
        ciudad = random.choices(
            list(ciudades_prov.keys()),
            weights=list(ciudades_prov.values())
        )[0]
        
        # Generar coordenadas
        lat, lon = generar_coordenadas(ciudad)
        
        # Generar fecha aleatoria
        dias_diff = (fecha_fin - fecha_inicio).days
        fecha_random = fecha_inicio + timedelta(days=random.randint(0, dias_diff))
        
        # Crear registro
        registro = {
            "id": i,
            "fecha": fecha_random.strftime("%Y-%m-%d"),
            "hora": f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}",
            "tipo_accidente": random.choice(TIPOS_ACCIDENTE),
            "provincia": provincia,
            "ciudad": ciudad,
            "direccion": generar_direccion(ciudad, provincia),
            "latitud": lat,
            "longitud": lon,
            "estado": random.choice(ESTADOS),
            "victimas": random.choices([0, 1, 2, 3, 4, 5], weights=[20, 40, 25, 10, 3, 2])[0],
            "heridos": random.choices([0, 1, 2, 3, 4], weights=[30, 35, 20, 10, 5])[0],
            "vehiculos_involucrados": random.choices([1, 2, 3, 4], weights=[30, 50, 15, 5])[0],
            "condicion_clima": random.choices(
                ["Despejado", "Cubierto", "Lluvia", "Neblina"],
                weights=[40, 35, 20, 5]
            )[0],
            "condicion_via": random.choices(
                ["Buena", "Regular", "Mala"],
                weights=[50, 35, 15]
            )[0]
        }
        
        datos.append(registro)
    
    return pd.DataFrame(datos)

if __name__ == "__main__":
    print("Generando dataset de 10,000 accidentes de tránsito...")
    print("70% en Azuay (85% de estos en Cuenca)")
    print("")
    
    df = generar_dataset(10000)
    
    # Guardar CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"accidentes_transito_ecuador_{len(df)}_{timestamp}.csv"
    df.to_csv(nombre_archivo, index=False, encoding='utf-8')
    
    print(f"✓ Archivo generado: {nombre_archivo}")
    print(f"✓ Total de registros: {len(df)}")
    print("")
    print("Distribución por provincia:")
    print(df['provincia'].value_counts())
    print("")
    print("Distribución en Azuay por ciudad:")
    print(df[df['provincia'] == 'Azuay']['ciudad'].value_counts())
    print("")
    print("Distribución de tipos de accidente:")
    print(df['tipo_accidente'].value_counts())
    print("")
    print("Rango de fechas:")
    print(f"Desde: {df['fecha'].min()}")
    print(f"Hasta: {df['fecha'].max()}")