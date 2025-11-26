# =======================================================================
# Integrated Smart Farm FastAPI Application (Final Robust Version)
# All Names and Comments are in English.
# =======================================================================

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine, Column, Integer, Float, Boolean, String, DateTime, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timedelta
from typing import List, Dict, Any
from pydantic import BaseModel, Field 

# ----------------------------------------------------------------------
# 1. Database Configuration
# ----------------------------------------------------------------------

# Using SQLite for immediate runnability.
SQLALCHEMY_DATABASE_URL = "sqlite:///./smart_farm.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    # Dependency: Provides a database session for each request
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ----------------------------------------------------------------------
# 2. ORM Models (Tables)
# ----------------------------------------------------------------------
# These models define the actual database structure.

class DBSensorData(Base):
    """Represents the sensor data table."""
    __tablename__ = "sensor_data"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    temperature = Column(Float, index=True)
    humidity = Column(Float)
    soil_moisture = Column(Float)
    irrigation_status = Column(Boolean)
    ai_decision = Column(String)

class DBPestReport(Base):
    """Represents the pest detection reports table."""
    __tablename__ = "pest_reports"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    pest_name = Column(String, index=True)
    plant_name = Column(String)
    detection_certainty = Column(Float)
    recommendation = Column(String)

class DBManualControl(Base): 
    """Stores the manual control state requested by the user."""
    __tablename__ = "manual_control"
    id = Column(Integer, primary_key=True, index=True, default=1) 
    manual_enabled = Column(Boolean, default=False) 
    pump_command = Column(Boolean, default=False) 
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ----------------------------------------------------------------------
# 3. Pydantic Models (Schemas)
# ----------------------------------------------------------------------
# These models define the data structure for API requests and responses.

# --- Sensor Data Schemas ---
class SensorDataBase(BaseModel):
    temperature: float
    humidity: float
    soil_moisture: float
    irrigation_status: bool
    ai_decision: str

class SensorDataCreate(SensorDataBase):
    pass

class SensorData(SensorDataBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True

# --- Pest Report Schemas ---
class PestReportBase(BaseModel):
    pest_name: str
    plant_name: str
    detection_certainty: float
    recommendation: str

class PestReportCreate(PestReportBase):
    pass

class PestReport(PestReportBase):
    id: int
    timestamp: datetime
    class Config:
        from_attributes = True

# --- Control Schemas ---
class ManualControlUpdate(BaseModel): 
    manual_enabled: bool = Field(..., description="Enable or disable manual control mode.")
    pump_command: bool = Field(..., description="Manual pump command (True=ON, False=OFF).")

class ManualControlStatus(BaseModel): 
    manual_enabled: bool
    pump_command: bool
    timestamp: datetime

    class Config:
        from_attributes = True

# --- Analytics Schemas ---
class WeeklyStatistics(BaseModel):
    week_start: str
    week_end: str
    avg_temperature: float
    avg_humidity: float
    avg_soil_moisture: float
    total_pest_reports: int


# ----------------------------------------------------------------------
# 4. CRUD Operations (Business Logic)
# ----------------------------------------------------------------------
# All functions interacting directly with the DB.

def get_manual_control(db: Session) -> DBManualControl:
    """Retrieves the manual control status (used by RPi)."""
    status = db.query(DBManualControl).filter(DBManualControl.id == 1).first()
    if not status:
        # Initialize default record
        status = DBManualControl(id=1, manual_enabled=False, pump_command=False)
        db.add(status)
        db.commit()
    return status

def update_manual_control(db: Session, update: ManualControlUpdate) -> DBManualControl:
    """Updates the manual control status (used by Frontend)."""
    status = get_manual_control(db)
    status.manual_enabled = update.manual_enabled
    status.pump_command = update.pump_command
    db.commit()
    db.refresh(status)
    return status

def create_sensor_data(db: Session, data: SensorDataCreate) -> DBSensorData:
    """Creates a new sensor data record."""
    db_data = DBSensorData(**data.model_dump())
    db.add(db_data)
    db.commit()
    db.refresh(db_data)
    return db_data

def get_latest_sensor_data(db: Session) -> DBSensorData | None:
    """Gets the latest sensor data record."""
    return db.query(DBSensorData).order_by(DBSensorData.timestamp.desc()).first()

def create_pest_report(db: Session, report: PestReportCreate) -> DBPestReport:
    """Creates a new pest report record."""
    db_report = DBPestReport(**report.model_dump())
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    return db_report

def get_recent_pest_reports(db: Session, limit: int = 10) -> List[DBPestReport]:
    """Gets the latest pest reports."""
    return db.query(DBPestReport).order_by(DBPestReport.timestamp.desc()).limit(limit).all()

def get_weekly_statistics(db: Session) -> List[WeeklyStatistics]:
    """Calculates aggregated weekly statistics for the last 4 weeks using pure Python logic."""
    now = datetime.now()
    four_weeks_ago = now - timedelta(weeks=4)
    sensor_records = db.query(DBSensorData).filter(DBSensorData.timestamp >= four_weeks_ago).all()
    pest_reports = db.query(DBPestReport).filter(DBPestReport.timestamp >= four_weeks_ago).all()
    if not sensor_records and not pest_reports:
        return []
    
    weekly_data: Dict[str, Dict[str, Any]] = {}

    def get_week_start(dt: datetime) -> datetime:
        """Determines the start of the week (Monday=0) for the given date."""
        day_of_week = dt.weekday()
        start = dt - timedelta(days=day_of_week)
        return start.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Process sensor data
    for record in sensor_records:
        week_start_dt = get_week_start(record.timestamp)
        week_key = week_start_dt.strftime('%Y-%m-%d')
        if week_key not in weekly_data:
            weekly_data[week_key] = {'count': 0, 'temps': [], 'hums': [], 'soils': [], 'pest_count': 0, 'week_start_dt': week_start_dt}
        weekly_data[week_key]['count'] += 1
        weekly_data[week_key]['temps'].append(record.temperature)
        weekly_data[week_key]['hums'].append(record.humidity)
        weekly_data[week_key]['soils'].append(record.soil_moisture)
    
    # Process pest reports
    for report in pest_reports:
        week_start_dt = get_week_start(report.timestamp)
        week_key = week_start_dt.strftime('%Y-%m-%d')
        if week_key not in weekly_data:
             weekly_data[week_key] = {'count': 0, 'temps': [], 'hums': [], 'soils': [], 'pest_count': 0, 'week_start_dt': week_start_dt}
        weekly_data[week_key]['pest_count'] += 1
    
    # Calculate averages and format the result
    results: List[WeeklyStatistics] = []
    sorted_weeks = sorted(weekly_data.keys())
    
    for week_key in sorted_weeks:
        data = weekly_data[week_key]
        if data['count'] > 0:
            avg_temp = sum(data['temps']) / data['count']
            avg_hum = sum(data['hums']) / data['count']
            avg_soil = sum(data['soils']) / data['count']
        else:
            avg_temp, avg_hum, avg_soil = 0.0, 0.0, 0.0
        
        week_start_dt = data['week_start_dt']
        week_end_dt = week_start_dt + timedelta(days=6)
        results.append(WeeklyStatistics(
            week_start=week_key,
            week_end=week_end_dt.strftime('%Y-%m-%d'),
            avg_temperature=round(avg_temp, 2),
            avg_humidity=round(avg_hum, 2),
            avg_soil_moisture=round(avg_soil, 2),
            total_pest_reports=data['pest_count']
        ))
    return results
    
# ----------------------------------------------------------------------
# 5. FastAPI Application (Endpoints)
# ----------------------------------------------------------------------

# Create tables (including DBManualControl) on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Smart Farm Backend API",
    description="API for managing smart farm data and manual control.",
    version="1.0.0",
)

# ----------------- Control Paths -----------------

@app.put("/control/manual/", response_model=ManualControlStatus, tags=["Control"])
def set_manual_control(update: ManualControlUpdate, db: Session = Depends(get_db)):
    """
    **[For Frontend]** Enable/Disable manual control mode and send pump command.
    """
    return update_manual_control(db, update)

@app.get("/control/status/", response_model=ManualControlStatus, tags=["Control"])
def get_manual_control_status(db: Session = Depends(get_db)):
    """
    **[For Raspberry Pi]** Retrieve the manual control status and requested command.
    """
    return get_manual_control(db)


# ----------------- Data Ingestion & Status Paths -----------------

@app.post("/data/sensor/", response_model=SensorData, status_code=201, tags=["Data Ingestion"])
def record_sensor_data(data: SensorDataCreate, db: Session = Depends(get_db)):
    """
    **Record Sensor Data**
    Used by Raspberry Pi to send temperature, humidity, soil moisture, 
    irrigation status, and current AI decision.
    """
    return create_sensor_data(db=db, data=data)

@app.post("/data/pest-report/", response_model=PestReport, status_code=201, tags=["Data Ingestion"])
def record_pest_report(report: PestReportCreate, db: Session = Depends(get_db)):
    """
    **Record Pest Report**
    Used to send detected pest reports (e.g., White Rot in Tomato) 
    along with necessary recommendations for the farmer.
    """
    return create_pest_report(db=db, report=report)

@app.get("/status/latest/", response_model=SensorData, tags=["Status & History"])
def get_current_status(db: Session = Depends(get_db)):
    """
    **Current Farm Status**
    Retrieves the latest record of sensor data and AI decision.
    """
    latest_data = get_latest_sensor_data(db)
    if latest_data is None:
        raise HTTPException(status_code=404, detail="No sensor data found.")
    return latest_data

@app.get("/reports/recent/", response_model=List[PestReport], tags=["Status & History"])
def get_latest_pest_reports(db: Session = Depends(get_db)):
    """
    **Latest Pest Reports**
    Retrieves a list of the most recently logged pest reports.
    """
    return get_recent_pest_reports(db)

@app.get("/statistics/weekly/", response_model=List[WeeklyStatistics], tags=["Analytics"])
def get_weekly_stats(db: Session = Depends(get_db)):
    """
    **Weekly Statistics**
    Retrieves weekly averages for temperature, humidity, and soil moisture, 
    and the total pest reports for each week (for the last 4 weeks).
    """
    return get_weekly_statistics(db)
