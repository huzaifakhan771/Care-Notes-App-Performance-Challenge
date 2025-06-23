"""
Care Notes API Runner

This script starts the FastAPI server for the Care Notes application.
It provides an API endpoint for retrieving care statistics with the optimized implementation.

Usage:
    python run_api.py

API Endpoints:
    GET /api/care-stats/
        Parameters:
            - tenant_id: ID of the tenant to get statistics for
            - facility_ids: (optional) Comma-separated list of facility IDs
            - date: (optional) Date in YYYY-MM-DD format (defaults to today)
"""

import uvicorn
from app import app, Base, create_test_data
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import asyncio

async def setup_database():
    """Initialize the database with test data if needed."""
    print("Setting up database...")
    engine = create_engine("sqlite:///care_notes.db", echo=False)
    Base.metadata.create_all(engine)
    
    # Create session
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        # Check if we need to create test data
        from sqlalchemy import text
        result = db.execute(text("SELECT COUNT(*) FROM care_notes")).scalar()
        
        if result == 0:
            print("No data found. Creating test data...")
            await create_test_data(db)
            print("Test data created successfully!")
        else:
            print(f"Database already contains {result} records.")
        
    finally:
        db.close()

if __name__ == "__main__":
    # Setup database with test data
    asyncio.run(setup_database())
    
    # Start the API server
    print("\n=== Starting Care Notes API Server ===")
    print("API documentation available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
