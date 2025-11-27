# Database Access Guide

This guide shows you how to access and query the SQLite database used by the Document Q&A system.

## Database Location

The database file is located at: `./documents.db` (in the project root)

## Method 1: Using the Database Viewer Script (Recommended)

We've created a Python script to easily view and query the database:

### Interactive Mode
```bash
python view_database.py
```

This will show an interactive menu where you can:
- View all tables
- View users
- View documents
- View queries
- View statistics
- Run custom SQL queries

### Command Line Mode

```bash
# View all tables
python view_database.py tables

# View users (limit 10)
python view_database.py users 10

# View documents for a specific user
python view_database.py documents default 20

# View all documents
python view_database.py documents 20

# View query history for a user
python view_database.py queries default 20

# View database statistics
python view_database.py stats

# Run custom SQL query
python view_database.py query "SELECT * FROM users LIMIT 5"
```

## Method 2: Using SQLite Command Line

### Access SQLite directly:
```bash
sqlite3 documents.db
```

### Useful SQLite Commands:

```sql
-- View all tables
.tables

-- View schema of a table
.schema users
.schema documents
.schema query_history

-- View all users
SELECT * FROM users;

-- View all documents
SELECT * FROM documents;

-- View query history
SELECT * FROM query_history;

-- Get statistics
SELECT COUNT(*) as total_users FROM users;
SELECT COUNT(*) as total_documents FROM documents;
SELECT COUNT(*) as total_queries FROM query_history;

-- View documents for a specific user
SELECT * FROM documents WHERE user_id = 'default';

-- View recent queries
SELECT question, answer, query_date 
FROM query_history 
ORDER BY query_date DESC 
LIMIT 10;

-- Exit SQLite
.quit
```

## Method 3: Using Python Scripts

### Example: Query database programmatically

```python
from database import get_db, get_user_documents, get_user_stats
from sqlalchemy.orm import Session

# Get database session
db = next(get_db())

# Get user documents
documents = get_user_documents(db, "default")
for doc in documents:
    print(f"{doc.filename}: {doc.character_count} characters")

# Get user statistics
stats = get_user_stats(db, "default")
print(stats)
```

## Method 4: Using API Endpoints

The backend provides REST API endpoints to access database information:

```bash
# Get user statistics
curl http://localhost:8000/users/default/stats

# Get user documents
curl http://localhost:8000/users/default/documents

# Get user query history
curl http://localhost:8000/users/default/queries

# Get overall database statistics
curl http://localhost:8000/db/stats
```

## Database Schema

### Users Table
- `id`: Integer (Primary Key)
- `user_id`: String (Unique, Indexed)
- `created_at`: DateTime
- `last_active`: DateTime

### Documents Table
- `id`: Integer (Primary Key)
- `user_id`: String (Indexed)
- `filename`: String
- `file_type`: String (pdf, txt, docx)
- `file_size`: Integer (bytes)
- `character_count`: Integer
- `upload_date`: DateTime
- `status`: String (processed, error, pending)

### Query History Table
- `id`: Integer (Primary Key)
- `user_id`: String (Indexed)
- `question`: Text
- `answer`: Text
- `query_date`: DateTime
- `response_time`: Float (seconds)

## Common Queries

### Find all users
```sql
SELECT user_id, created_at, last_active FROM users;
```

### Find documents by user
```sql
SELECT filename, file_type, upload_date, character_count 
FROM documents 
WHERE user_id = 'default';
```

### Find recent queries
```sql
SELECT question, query_date, response_time 
FROM query_history 
ORDER BY query_date DESC 
LIMIT 10;
```

### Get user statistics
```sql
SELECT 
    user_id,
    COUNT(DISTINCT documents.id) as doc_count,
    COUNT(DISTINCT query_history.id) as query_count,
    SUM(documents.character_count) as total_chars
FROM users
LEFT JOIN documents ON users.user_id = documents.user_id
LEFT JOIN query_history ON users.user_id = query_history.user_id
GROUP BY users.user_id;
```

## Backup Database

```bash
# Create backup
cp documents.db documents_backup.db

# Or use SQLite backup command
sqlite3 documents.db ".backup documents_backup.db"
```

## Notes

- The database is SQLite, so it's a single file
- All data persists across server restarts
- The database is automatically created on first run
- User data is isolated per `user_id`

