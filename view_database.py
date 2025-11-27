"""
Database viewer and query tool for the document Q&A system.
"""
import sqlite3
from database import DATABASE_URL, init_db
import sys
from tabulate import tabulate


def view_all_tables():
    """View all tables in the database."""
    conn = sqlite3.connect(DATABASE_URL.replace("sqlite:///", ""))
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print("\n📊 Database Tables:")
    print("=" * 50)
    for table in tables:
        print(f"  • {table[0]}")
    print()
    
    conn.close()


def view_users(limit=10):
    """View users in the database."""
    conn = sqlite3.connect(DATABASE_URL.replace("sqlite:///", ""))
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT user_id, created_at, last_active 
        FROM users 
        ORDER BY last_active DESC 
        LIMIT ?
    """, (limit,))
    
    users = cursor.fetchall()
    
    if users:
        print(f"\n👤 Users (showing {len(users)}):")
        print("=" * 80)
        print(tabulate(users, headers=["User ID", "Created At", "Last Active"], tablefmt="grid"))
    else:
        print("\n👤 No users found in database.")
    
    conn.close()


def view_documents(user_id=None, limit=20):
    """View documents in the database."""
    conn = sqlite3.connect(DATABASE_URL.replace("sqlite:///", ""))
    cursor = conn.cursor()
    
    if user_id:
        cursor.execute("""
            SELECT id, user_id, filename, file_type, file_size, character_count, upload_date, status
            FROM documents 
            WHERE user_id = ?
            ORDER BY upload_date DESC 
            LIMIT ?
        """, (user_id, limit))
    else:
        cursor.execute("""
            SELECT id, user_id, filename, file_type, file_size, character_count, upload_date, status
            FROM documents 
            ORDER BY upload_date DESC 
            LIMIT ?
        """, (limit,))
    
    documents = cursor.fetchall()
    
    if documents:
        print(f"\n📄 Documents (showing {len(documents)}):")
        print("=" * 120)
        print(tabulate(
            documents,
            headers=["ID", "User ID", "Filename", "Type", "Size (bytes)", "Chars", "Upload Date", "Status"],
            tablefmt="grid"
        ))
    else:
        print("\n📄 No documents found in database.")
    
    conn.close()


def view_queries(user_id=None, limit=20):
    """View query history in the database."""
    conn = sqlite3.connect(DATABASE_URL.replace("sqlite:///", ""))
    cursor = conn.cursor()
    
    if user_id:
        cursor.execute("""
            SELECT id, user_id, question, answer, query_date, response_time
            FROM query_history 
            WHERE user_id = ?
            ORDER BY query_date DESC 
            LIMIT ?
        """, (user_id, limit))
    else:
        cursor.execute("""
            SELECT id, user_id, question, answer, query_date, response_time
            FROM query_history 
            ORDER BY query_date DESC 
            LIMIT ?
        """, (limit,))
    
    queries = cursor.fetchall()
    
    if queries:
        print(f"\n💬 Query History (showing {len(queries)}):")
        print("=" * 120)
        # Truncate long answers for display
        display_queries = []
        for q in queries:
            answer_preview = q[3][:100] + "..." if len(q[3]) > 100 else q[3]
            display_queries.append((q[0], q[1], q[2][:50] + "..." if len(q[2]) > 50 else q[2], answer_preview, q[4], q[5]))
        
        print(tabulate(
            display_queries,
            headers=["ID", "User ID", "Question", "Answer (preview)", "Date", "Response Time (s)"],
            tablefmt="grid"
        ))
    else:
        print("\n💬 No queries found in database.")
    
    conn.close()


def get_stats():
    """Get database statistics."""
    conn = sqlite3.connect(DATABASE_URL.replace("sqlite:///", ""))
    cursor = conn.cursor()
    
    # Count users
    cursor.execute("SELECT COUNT(*) FROM users")
    user_count = cursor.fetchone()[0]
    
    # Count documents
    cursor.execute("SELECT COUNT(*) FROM documents")
    doc_count = cursor.fetchone()[0]
    
    # Count queries
    cursor.execute("SELECT COUNT(*) FROM query_history")
    query_count = cursor.fetchone()[0]
    
    # Total characters
    cursor.execute("SELECT SUM(character_count) FROM documents")
    total_chars = cursor.fetchone()[0] or 0
    
    # Average response time
    cursor.execute("SELECT AVG(response_time) FROM query_history WHERE response_time IS NOT NULL")
    avg_response = cursor.fetchone()[0] or 0
    
    print("\n📊 Database Statistics:")
    print("=" * 50)
    print(f"  Users: {user_count}")
    print(f"  Documents: {doc_count}")
    print(f"  Queries: {query_count}")
    print(f"  Total Characters: {total_chars:,}")
    print(f"  Avg Response Time: {avg_response:.2f}s")
    print()
    
    conn.close()


def run_custom_query(query):
    """Run a custom SQL query."""
    conn = sqlite3.connect(DATABASE_URL.replace("sqlite:///", ""))
    cursor = conn.cursor()
    
    try:
        cursor.execute(query)
        
        # Check if it's a SELECT query
        if query.strip().upper().startswith("SELECT"):
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            if results:
                print(f"\n📋 Query Results ({len(results)} rows):")
                print("=" * 100)
                print(tabulate(results, headers=columns, tablefmt="grid"))
            else:
                print("\n📋 No results found.")
        else:
            conn.commit()
            print(f"\n✅ Query executed successfully. Rows affected: {cursor.rowcount}")
    except Exception as e:
        print(f"\n❌ Error executing query: {e}")
    
    conn.close()


def main():
    """Main menu for database viewer."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "tables":
            view_all_tables()
        elif command == "users":
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            view_users(limit)
        elif command == "documents":
            user_id = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].isdigit() else None
            limit = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else int(sys.argv[3]) if len(sys.argv) > 3 else 20
            view_documents(user_id, limit)
        elif command == "queries":
            user_id = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].isdigit() else None
            limit = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else int(sys.argv[3]) if len(sys.argv) > 3 else 20
            view_queries(user_id, limit)
        elif command == "stats":
            get_stats()
        elif command == "query":
            if len(sys.argv) > 2:
                query = " ".join(sys.argv[2:])
                run_custom_query(query)
            else:
                print("❌ Please provide a SQL query.")
        else:
            print_help()
    else:
        # Interactive mode
        print("\n" + "=" * 60)
        print("🗄️  Database Viewer - Document Q&A System")
        print("=" * 60)
        
        while True:
            print("\nOptions:")
            print("  1. View all tables")
            print("  2. View users")
            print("  3. View documents")
            print("  4. View queries")
            print("  5. View statistics")
            print("  6. Run custom query")
            print("  7. Exit")
            
            choice = input("\nEnter choice (1-7): ").strip()
            
            if choice == "1":
                view_all_tables()
            elif choice == "2":
                limit = input("Limit (default 10): ").strip()
                limit = int(limit) if limit else 10
                view_users(limit)
            elif choice == "3":
                user_id = input("User ID (optional, press Enter for all): ").strip()
                user_id = user_id if user_id else None
                limit = input("Limit (default 20): ").strip()
                limit = int(limit) if limit else 20
                view_documents(user_id, limit)
            elif choice == "4":
                user_id = input("User ID (optional, press Enter for all): ").strip()
                user_id = user_id if user_id else None
                limit = input("Limit (default 20): ").strip()
                limit = int(limit) if limit else 20
                view_queries(user_id, limit)
            elif choice == "5":
                get_stats()
            elif choice == "6":
                query = input("Enter SQL query: ").strip()
                if query:
                    run_custom_query(query)
            elif choice == "7":
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please try again.")


def print_help():
    """Print help message."""
    print("""
🗄️  Database Viewer - Usage:

  python view_database.py [command] [options]

Commands:
  tables                    - View all tables
  users [limit]             - View users (default limit: 10)
  documents [user_id] [limit] - View documents (default limit: 20)
  queries [user_id] [limit]  - View query history (default limit: 20)
  stats                     - View database statistics
  query "SQL QUERY"         - Run custom SQL query

Examples:
  python view_database.py tables
  python view_database.py users 20
  python view_database.py documents default 10
  python view_database.py queries default
  python view_database.py stats
  python view_database.py query "SELECT * FROM users LIMIT 5"

Interactive Mode:
  python view_database.py
    (Run without arguments for interactive menu)
""")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        print_help()
    else:
        main()

