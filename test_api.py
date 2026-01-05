import httpx
import asyncio
import json

async def test_api():
    """
    Script de prueba para verificar que la API externa devuelve datos
    """
    API_URL = "http://localhost:8080/api/accidentes/cercanos"
    
    params = {
        "lat": -2.89264,
        "lon": -78.77814
    }
    
    print("🔍 Probando conexión con API externa...")
    print(f"URL: {API_URL}")
    print(f"Parámetros: {params}\n")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(API_URL, params=params)
            
            print(f"✅ Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Total accidentes recibidos: {len(data)}\n")
                
                if len(data) > 0:
                    print("📋 Ejemplo de primer accidente:")
                    print(json.dumps(data[0], indent=2, ensure_ascii=False))
                    
                    print("\n🔍 Analizando estructura de datos:")
                    primer_accidente = data[0]
                    
                    # Verificar tipo de accidente
                    if 'tipoaccidente' in primer_accidente:
                        tipo = primer_accidente['tipoaccidente']
                        print(f"  ✓ Tipo de accidente: {tipo.get('nombre', 'N/A')}")
                    else:
                        print("  ✗ No se encontró 'tipoaccidente'")
                    
                    # Verificar ruta y ubicación
                    if 'ruta' in primer_accidente and primer_accidente['ruta']:
                        ruta = primer_accidente['ruta']
                        if 'ciudad' in ruta and ruta['ciudad']:
                            ciudad = ruta['ciudad']
                            print(f"  ✓ Ciudad: {ciudad.get('nombreCiudad', 'N/A')}")
                            
                            if 'provincia' in ciudad and ciudad['provincia']:
                                provincia = ciudad['provincia']
                                print(f"  ✓ Provincia: {provincia.get('nombreProvincia', 'N/A')}")
                            else:
                                print("  ✗ No se encontró 'provincia'")
                        else:
                            print("  ✗ No se encontró 'ciudad'")
                    else:
                        print("  ✗ No se encontró 'ruta'")
                    
                    print("\n📊 Resumen de todos los accidentes:")
                    tipos = {}
                    provincias = {}
                    
                    for acc in data:
                        # Contar tipos
                        if 'tipoaccidente' in acc and acc['tipoaccidente']:
                            nombre_tipo = acc['tipoaccidente'].get('nombre', 'Desconocido')
                            tipos[nombre_tipo] = tipos.get(nombre_tipo, 0) + 1
                        
                        # Contar provincias
                        if 'ruta' in acc and acc['ruta'] and 'ciudad' in acc['ruta']:
                            ciudad = acc['ruta']['ciudad']
                            if 'provincia' in ciudad and ciudad['provincia']:
                                nombre_prov = ciudad['provincia'].get('nombreProvincia', 'Desconocido')
                                provincias[nombre_prov] = provincias.get(nombre_prov, 0) + 1
                    
                    print("\n  Tipos de accidentes encontrados:")
                    for tipo, cantidad in tipos.items():
                        print(f"    - {tipo}: {cantidad}")
                    
                    print("\n  Provincias encontradas:")
                    for prov, cantidad in provincias.items():
                        print(f"    - {prov}: {cantidad}")
                else:
                    print("⚠️ La API devolvió una lista vacía")
            else:
                print(f"❌ Error: La API devolvió código {response.status_code}")
                print(f"Respuesta: {response.text}")
                
    except httpx.ConnectError:
        print("❌ Error: No se pudo conectar a la API")
        print("   Verifica que el servidor esté corriendo en http://localhost:8080")
    except httpx.TimeoutException:
        print("❌ Error: Timeout al conectar con la API")
    except Exception as e:
        print(f"❌ Error inesperado: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_api())