import requests
import json

# Configuración
API_URL = "http://localhost:8000"
CSV_FILE = "accidentes_transito_ecuador_10000.csv"

def test_analizar_accidentes():
    """Prueba el endpoint de análisis"""
    print("=" * 60)
    print("🧪 Probando análisis de accidentes...")
    print("=" * 60)
    
    # Parámetros
    params = {
        'latitud': -2.89264,
        'longitud': -78.77814,
        'radio_km': 10
    }
    
    # Abrir archivo
    try:
        with open(CSV_FILE, 'rb') as f:
            files = {'archivo': (CSV_FILE, f, 'text/csv')}
            data = {
                'latitud': params['latitud'],
                'longitud': params['longitud'],
                'radio_km': params['radio_km']
            }
            
            print(f"\n📤 Enviando solicitud a: {API_URL}/analizar")
            print(f"📍 Punto de referencia: {params['latitud']}, {params['longitud']}")
            print(f"📏 Radio de búsqueda: {params['radio_km']} km")
            print("\n⏳ Procesando...\n")
            
            response = requests.post(
                f"{API_URL}/analizar",
                files=files,
                data=data,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print("✅ Análisis completado exitosamente!\n")
                print("=" * 60)
                print("📊 ESTADÍSTICAS GENERALES")
                print("=" * 60)
                
                stats = data['estadisticas']
                print(f"📁 Total accidentes CSV: {stats['total_csv']}")
                print(f"🌐 Total accidentes API: {stats['total_api']}")
                print(f"📈 Total combinado: {stats['total_combinado']}")
                print(f"🎯 Accidentes en radio: {stats['accidentes_en_radio']}")
                
                print("\n" + "=" * 60)
                print("🚗 TIPOS DE ACCIDENTES MÁS COMUNES")
                print("=" * 60)
                for tipo, cantidad in list(stats['tipos_mas_comunes'].items())[:5]:
                    print(f"  • {tipo}: {cantidad}")
                
                print("\n" + "=" * 60)
                print("🗺️  PROVINCIAS AFECTADAS")
                print("=" * 60)
                for provincia, cantidad in stats['provincias_afectadas'].items():
                    print(f"  • {provincia}: {cantidad}")
                
                if data['zonas_peligrosas']:
                    print("\n" + "=" * 60)
                    print("⚠️  ZONAS PELIGROSAS IDENTIFICADAS")
                    print("=" * 60)
                    for i, zona in enumerate(data['zonas_peligrosas'][:5], 1):
                        print(f"\n  Zona {i}:")
                        print(f"    📍 Ubicación: {zona['latitud']:.6f}, {zona['longitud']:.6f}")
                        print(f"    💥 Accidentes: {zona['cantidad_accidentes']}")
                        print(f"    🔴 Nivel: {zona['nivel_peligro']}")
                        print(f"    📏 Radio: {zona['radio_metros']}m")
                
                print("\n" + "=" * 60)
                print("💡 RECOMENDACIONES")
                print("=" * 60)
                for rec in data['recomendaciones']:
                    print(f"  {rec}")
                
                print("\n" + "=" * 60)
                print("✅ Prueba completada exitosamente!")
                print("=" * 60)
                
                # Guardar resultado completo
                with open('resultado_analisis.json', 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print("\n💾 Resultado completo guardado en: resultado_analisis.json")
                
            else:
                print(f"❌ Error {response.status_code}")
                print(response.json())
                
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{CSV_FILE}'")
        print("💡 Ejecuta primero: python generar_csv.py")
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: No se puede conectar a {API_URL}")
        print("💡 Asegúrate de que la API esté corriendo:")
        print("   uvicorn main:app --reload")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

def test_api_externa():
    """Prueba la conexión con la API externa"""
    print("\n" + "=" * 60)
    print("🧪 Probando conexión con API externa...")
    print("=" * 60)
    
    try:
        response = requests.get(
            f"{API_URL}/api-externa/test",
            params={'lat': -2.89264, 'lon': -78.77814},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Conexión exitosa!")
            print(f"📊 Accidentes encontrados: {len(data['data'])}")
        else:
            print(f"❌ Error {response.status_code}")
            print(response.json())
            
    except Exception as e:
        print(f"⚠️  No se pudo conectar a la API externa: {e}")
        print("💡 Esto es normal si la API externa no está corriendo")

if __name__ == "__main__":
    print("\n")
    print("🚀 SISTEMA DE ANÁLISIS DE ACCIDENTES DE TRÁNSITO")
    print("=" * 60)
    
    # Probar API externa primero
    test_api_externa()
    
    # Probar análisis principal
    test_analizar_accidentes()
    
    print("\n🎉 Todas las pruebas completadas!\n")