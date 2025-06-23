"""
Care Notes Load Test

This script performs load testing on the Care Notes API to demonstrate
its ability to handle concurrent requests efficiently.

Usage:
    python load_test.py [--concurrency 50] [--duration 30]

Options:
    --concurrency: Number of concurrent users (default: 50)
    --duration: Test duration in seconds (default: 30)
"""

import asyncio
import aiohttp
import time
import random
import argparse
import json
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

# Parse command line arguments
parser = argparse.ArgumentParser(description='Load test for Care Notes API')
parser.add_argument('--concurrency', type=int, default=50, help='Number of concurrent users')
parser.add_argument('--duration', type=int, default=30, help='Test duration in seconds')
args = parser.parse_args()

# API configuration
API_URL = "http://localhost:8000/api/care-stats/"
TENANTS = list(range(1, 6))  # Tenants 1-5
FACILITIES = {tenant: list(range(1, 4)) for tenant in TENANTS}  # 3 facilities per tenant

# Generate test dates (last 30 days)
today = datetime.now()
TEST_DATES = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30)]

# Statistics tracking
response_times = []
status_counts = defaultdict(int)
error_messages = []
request_count = 0

async def make_request(session, tenant_id, facility_ids=None, date=None):
    """Make a request to the API with the given parameters."""
    global request_count
    
    params = {"tenant_id": tenant_id}
    
    if facility_ids:
        params["facility_ids"] = ",".join(map(str, facility_ids))
    
    if date:
        params["date"] = date
    
    start_time = time.time()
    
    try:
        async with session.get(API_URL, params=params) as response:
            status_counts[response.status] += 1
            
            if response.status == 200:
                await response.json()  # Ensure we read the response body
            else:
                error_text = await response.text()
                error_messages.append(f"Status {response.status}: {error_text}")
                
            elapsed = time.time() - start_time
            response_times.append(elapsed)
            request_count += 1
            
            if request_count % 100 == 0:
                print(f"Completed {request_count} requests")
                
    except Exception as e:
        error_messages.append(f"Request error: {str(e)}")
        status_counts["error"] += 1

async def user_session(session, user_id, end_time):
    """Simulate a user making repeated requests until the end time."""
    while time.time() < end_time:
        # Randomly select parameters
        tenant_id = random.choice(TENANTS)
        
        # 50% chance to include facility filtering
        facility_ids = None
        if random.random() > 0.5:
            num_facilities = random.randint(1, 3)
            facility_ids = random.sample(FACILITIES[tenant_id], num_facilities)
        
        # 30% chance to include date filtering
        date = None
        if random.random() > 0.7:
            date = random.choice(TEST_DATES)
        
        await make_request(session, tenant_id, facility_ids, date)
        
        # Small random delay between requests (10-100ms)
        await asyncio.sleep(random.uniform(0.01, 0.1))

async def run_load_test(concurrency, duration):
    """Run the load test with the specified concurrency and duration."""
    print(f"Starting load test with {concurrency} concurrent users for {duration} seconds")
    
    end_time = time.time() + duration
    
    # Create a shared session for all users
    async with aiohttp.ClientSession() as session:
        # Create user tasks
        tasks = [user_session(session, i, end_time) for i in range(concurrency)]
        
        # Run all tasks concurrently
        await asyncio.gather(*tasks)

def print_results():
    """Print the load test results."""
    print("\n=== Load Test Results ===")
    print(f"Total requests: {request_count}")
    
    if response_times:
        avg_time = statistics.mean(response_times)
        median_time = statistics.median(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        p95_time = sorted(response_times)[int(len(response_times) * 0.95)]
        
        print(f"\nResponse Time Statistics:")
        print(f"  Average: {avg_time:.4f} seconds")
        print(f"  Median: {median_time:.4f} seconds")
        print(f"  Min: {min_time:.4f} seconds")
        print(f"  Max: {max_time:.4f} seconds")
        print(f"  95th percentile: {p95_time:.4f} seconds")
        
        print(f"\nRequests per second: {request_count / args.duration:.2f}")
    
    print("\nStatus Code Distribution:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count} ({count/request_count*100:.1f}%)")
    
    if error_messages:
        print("\nSample Errors:")
        for error in error_messages[:5]:  # Show at most 5 errors
            print(f"  {error}")
        
        if len(error_messages) > 5:
            print(f"  ... and {len(error_messages) - 5} more errors")

if __name__ == "__main__":
    # Ensure the API is running
    print("Make sure the API server is running (python run_api.py)")
    print("Press Enter to start the load test...")
    input()
    
    # Run the load test
    start_time = time.time()
    asyncio.run(run_load_test(args.concurrency, args.duration))
    total_time = time.time() - start_time
    
    # Print results
    print_results()
