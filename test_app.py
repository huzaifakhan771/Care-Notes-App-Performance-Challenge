"""
Care Notes App Tests

This module contains tests for the Care Notes application,
focusing on the optimized query implementation.
"""

import unittest
import asyncio
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app import Base, CareNote, get_daily_care_stats, get_daily_care_stats_optimized, create_test_data

class TestCareNotesApp(unittest.TestCase):
    """Test cases for the Care Notes application."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test database and create test data."""
        # Create in-memory SQLite database for testing
        cls.engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(cls.engine)
        
        # Create session
        cls.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=cls.engine)
        cls.db = cls.SessionLocal()
        
        # Create test data
        cls.data_info = asyncio.run(create_test_data(cls.db))
        
        # Get test parameters
        cls.test_tenant_id = 3
        cls.test_date = datetime.utcnow().date()
        cls.test_facilities = cls.data_info["facilities"][cls.test_tenant_id]
    
    @classmethod
    def tearDownClass(cls):
        """Clean up resources."""
        cls.db.close()
    
    def test_results_consistency(self):
        """Test that original and optimized implementations return consistent results."""
        # Get results from both implementations
        original_result = asyncio.run(get_daily_care_stats(
            self.db, 
            tenant_id=self.test_tenant_id, 
            date=datetime.combine(self.test_date, datetime.min.time())
        ))
        
        optimized_result = asyncio.run(get_daily_care_stats_optimized(
            self.db, 
            tenant_id=self.test_tenant_id, 
            date=datetime.combine(self.test_date, datetime.min.time())
        ))
        
        # Remove query_time_ms from optimized result for comparison
        optimized_result_compare = {k: v for k, v in optimized_result.items() if k != 'query_time_ms'}
        
        # Check total notes count
        self.assertEqual(
            original_result["total_notes"], 
            optimized_result_compare["total_notes"],
            "Total notes count should be the same in both implementations"
        )
        
        # Check category keys
        self.assertEqual(
            set(original_result["by_category"].keys()), 
            set(optimized_result_compare["by_category"].keys()),
            "Category keys should be the same in both implementations"
        )
        
        # Check facility keys
        self.assertEqual(
            set(original_result["by_facility"].keys()), 
            set(optimized_result_compare["by_facility"].keys()),
            "Facility keys should be the same in both implementations"
        )
    
    def test_facility_filtering(self):
        """Test that facility filtering works correctly."""
        # Test with a subset of facilities
        test_facility_subset = self.test_facilities[:1]  # Just the first facility
        
        # Get results with facility filtering
        filtered_result = asyncio.run(get_daily_care_stats_optimized(
            self.db, 
            tenant_id=self.test_tenant_id,
            facility_ids=test_facility_subset,
            date=datetime.combine(self.test_date, datetime.min.time())
        ))
        
        # Check that only the specified facility is in the results
        self.assertEqual(
            set(filtered_result["by_facility"].keys()),
            set(test_facility_subset),
            "Facility filtering should only return data for the specified facilities"
        )
    
    def test_caching(self):
        """Test that caching works correctly."""
        # First call to populate cache
        start_time = datetime.now()
        first_result = asyncio.run(get_daily_care_stats_optimized(
            self.db, 
            tenant_id=self.test_tenant_id,
            date=datetime.combine(self.test_date, datetime.min.time())
        ))
        first_call_time = (datetime.now() - start_time).total_seconds()
        
        # Second call should use cache
        start_time = datetime.now()
        second_result = asyncio.run(get_daily_care_stats_optimized(
            self.db, 
            tenant_id=self.test_tenant_id,
            date=datetime.combine(self.test_date, datetime.min.time())
        ))
        second_call_time = (datetime.now() - start_time).total_seconds()
        
        # Check that second call is significantly faster
        self.assertLess(
            second_call_time,
            first_call_time / 10,  # At least 10x faster
            "Cached query should be significantly faster"
        )
        
        # Check that results are the same
        self.assertEqual(
            first_result["total_notes"],
            second_result["total_notes"],
            "Cached results should be the same as original results"
        )

if __name__ == "__main__":
    unittest.main()
