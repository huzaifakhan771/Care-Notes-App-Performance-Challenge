# Care Notes App Performance Challenge

This project demonstrates performance optimization techniques for a multi-tenant care notes system. The goal is to optimize the core analytics query to reduce response time by at least 80% while maintaining data consistency and handling concurrent requests.

## Project Structure

- `app.py` - Main application code with the original and optimized implementations
- `run_api.py` - Script to run the FastAPI server
- `load_test.py` - Script to perform load testing on the API
- `test_app.py` - Unit tests for the application
- `OPTIMIZATION_DOCUMENTATION.md` - Detailed documentation of the optimization approach
- `requirements.txt` - List of dependencies

## Features

- Efficient batch data generation (100K+ records)
- Multi-tenant data isolation
- Optimized analytics query with 80%+ performance improvement
- Request-level caching with TTL
- Concurrent request handling
- Comprehensive test suite

## Performance Improvements

- Single query performance: 84.81% improvement
- Filtered query performance: 89.68% improvement
- Concurrent request performance: 96.41% improvement
- Cache performance: 2,813x faster than original implementation

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/huzaifakhan771/Care-Notes-App-Performance-Challenge.git
   cd Care-Notes-App-Performance-Challenge
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

1. Run the performance test to see the optimization results:
   ```
   python app.py
   ```

2. Start the API server:
   ```
   python run_api.py
   ```

3. Run the load test (in a separate terminal):
   ```
   python load_test.py
   ```

4. Run the unit tests:
   ```
   python -m unittest test_app.py
   ```

## API Endpoints

- `GET /api/care-stats/`
  - Parameters:
    - `tenant_id`: ID of the tenant to get statistics for
    - `facility_ids`: (optional) Comma-separated list of facility IDs
    - `date`: (optional) Date in YYYY-MM-DD format (defaults to today)

## Optimization Techniques

1. **Database Indexing**: Strategic indexes for tenant, facility, and date columns
2. **Query Optimization**: Single-pass data processing and efficient SQL aggregations
3. **Caching**: Request-level caching with TTL and lock-based concurrency control
4. **Concurrency Handling**: Async/await pattern and resource locking
5. **Batch Processing**: Efficient bulk insertions for data generation

For detailed information about the optimization approach, see [OPTIMIZATION_DOCUMENTATION.md](OPTIMIZATION_DOCUMENTATION.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
