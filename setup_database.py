#!/usr/bin/env python3
"""
SQLite Database Setup Script for Multi-Face Attendance System

This script sets up the SQLite database and tables.
Run this before starting the application.
"""

import os
from utils.db_utils import create_tables
from config import DATABASE_URL

def create_sqlite_database():
    """Create the SQLite database file if it doesn't exist."""
    db_path = DATABASE_URL.replace("sqlite:///", "")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    if not os.path.exists(db_path):
        open(db_path, 'a').close()
        print(f"✅ SQLite database created at: {db_path}")
    else:
        print(f"ℹ️  SQLite database already exists at: {db_path}")

def setup_tables():
    """Create all tables using SQLAlchemy."""
    try:
        create_tables()
        print("✅ All tables created successfully!")
        return True
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False

def main():
    print("🚀 Setting up Multi-Face Attendance System (SQLite) Database...")
    print("-" * 60)

    print(f"Database URL: {DATABASE_URL}")
    print("-" * 60)

    # Create SQLite DB
    print("\n1. Creating SQLite database...")
    create_sqlite_database()

    # Create tables
    print("\n2. Creating tables...")
    if not setup_tables():
        print("❌ Failed to create tables. Please check your SQLAlchemy models.")
        return

    print("\n🎉 Database setup completed successfully (SQLite)!")
    print("\nNext steps:")
    print("1. Run: pip install -r requirements.txt")
    print("2. Start the app: streamlit run app.py")

if __name__ == "__main__":
    main()
