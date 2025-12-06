"""
Database migration script to add missing columns to existing tables.
"""
import sqlite3
from pathlib import Path

DATABASE_PATH = Path("./documents.db")

def migrate_database():
    """Add missing columns to the database tables."""
    if not DATABASE_PATH.exists():
        print("Database file not found. Run init_db() to create it.")
        return
    
    conn = sqlite3.connect(str(DATABASE_PATH))
    cursor = conn.cursor()
    
    try:
        print("🔧 Starting database migration...")
        
        # Check current schema
        cursor.execute("PRAGMA table_info(users)")
        existing_columns = [row[1] for row in cursor.fetchall()]
        print(f"   Existing columns in users table: {existing_columns}")
        
        # Columns to add if missing
        columns_to_add = [
            ("email", "VARCHAR", True),
            ("phone", "VARCHAR", True),
            ("auth_type", "VARCHAR", True),
            ("google_id", "VARCHAR", True),
            ("password_hash", "VARCHAR", True),
            ("is_verified", "VARCHAR", False),
        ]
        
        for column_name, column_type, nullable in columns_to_add:
            if column_name not in existing_columns:
                print(f"   Adding column: {column_name}")
                null_constraint = "" if nullable else " NOT NULL DEFAULT ''"
                default_value = "NULL" if nullable else "''"
                
                # For SQLite, we need to handle DEFAULT values differently
                if column_name == "is_verified":
                    default_value = "'false'"
                
                try:
                    cursor.execute(f"ALTER TABLE users ADD COLUMN {column_name} {column_type}{null_constraint} DEFAULT {default_value}")
                    print(f"   ✓ Added {column_name}")
                except sqlite3.OperationalError as e:
                    if "duplicate column" in str(e).lower():
                        print(f"   ⚠️ Column {column_name} already exists")
                    else:
                        raise
        
        # Commit changes
        conn.commit()
        print("✅ Database migration completed successfully!")
        
        # Show final schema
        cursor.execute("PRAGMA table_info(users)")
        final_columns = cursor.fetchall()
        print("\n📋 Final users table schema:")
        for col in final_columns:
            print(f"   {col[1]} ({col[2]})")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Migration failed: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_database()

