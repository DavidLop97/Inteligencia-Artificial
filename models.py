from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import date

class AccidenteRequest(BaseModel):
    latitud: float = Field(..., description="Latitud del punto de referencia")
    longitud: float = Field(..., description="Longitud del punto de referencia")
    radio_km: Optional[float] = Field(5.0, description="Radio de búsqueda en km")

class AccidenteCSV(BaseModel):
    id: int
    fecha: str
    tipo_accidente: str
    provincia: str
    ciudad: str
    direccion: str
    latitud: float
    longitud: float
    distancia_km: float
    estado: str

class AccidenteAPI(BaseModel):
    idAccidenteTransito: int
    fechaSiniestro: str
    tipoaccidente: Dict
    ruta: Dict
    latitud: float
    longitud: float
    direccion: str
    estado: str
    aprobado: bool
    distancia_km: Optional[float] = None

class EstadisticasGenerales(BaseModel):
    total_csv: int
    total_api: int
    total_combinado: int
    accidentes_en_radio: int
    tipos_mas_comunes: Dict[str, int]
    provincias_afectadas: Dict[str, int]

class ZonaPeligrosa(BaseModel):
    latitud: float
    longitud: float
    cantidad_accidentes: int
    radio_metros: float
    nivel_peligro: str

class CombinedResponse(BaseModel):
    punto_referencia: Dict[str, float]
    radio_busqueda_km: float
    estadisticas: EstadisticasGenerales
    accidentes_csv: List[AccidenteCSV]
    accidentes_api: List[AccidenteAPI]
    zonas_peligrosas: List[ZonaPeligrosa]
    recomendaciones: List[str]

class AccidenteResponse(BaseModel):
    success: bool
    message: str
    data: Optional[CombinedResponse] = None