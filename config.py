import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

# =========================================================
# Load environment variables
# =========================================================
load_dotenv()

# =========================================================
# ✅ SQLite Database Configuration (Replaces PostgreSQL)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "attendance_system.db")

# Ensure 'data' directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

DATABASE_CONFIG = {
    'url': f"sqlite:///{DB_PATH}"
}

# SQLAlchemy engine
engine = create_engine(DATABASE_CONFIG['url'], echo=False)

# Also expose DATABASE_URL for compatibility with setup_database.py
DATABASE_URL = DATABASE_CONFIG['url']

# =========================================================
# Face Recognition Configuration (Optimized for i7 12th Gen)
# =========================================================
FACE_RECOGNITION_CONFIG = {
    'tolerance': 0.6,
    'model': 'cnn',  # Use 'cnn' if GPU available, else fallback to 'hog'
    'min_confidence': 0.6,
    'max_faces_per_frame': 70,
    'face_detection_batch_size': 8,
    'encoding_cache_size': 1000,
    'use_gpu': True,
    'num_jitters': 1,
    'resize_factor': 0.25,
    'enable_enhancement': True,
    'anti_spoofing': True,
    'quality_threshold': 0.3,
    'max_detection_time': 10.0
}

# =========================================================
# Performance Optimization
# =========================================================
PERFORMANCE_CONFIG = {
    'max_workers': 8,
    'memory_limit_mb': 4096,
    'image_quality': 85,
    'max_image_size_mb': 5,
    'cache_ttl_seconds': 3600,
    'enable_logging': True,
    'log_level': 'INFO'
}

# =========================================================
# Authentication Configuration
# =========================================================
AUTH_CONFIG = {
    'session_timeout_hours': 8,
    'max_login_attempts': 3,
    'password_min_length': 8,
    'require_special_chars': True,
    'jwt_secret_key': os.getenv('JWT_SECRET', 'your-secret-key-change-this'),
    'cookie_expiry_days': 7
}

# =========================================================
# Email Configuration
# =========================================================
EMAIL_CONFIG = {
    'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
    'smtp_port': int(os.getenv('SMTP_PORT', '587')),
    'email': os.getenv('EMAIL_USER', 'your-email@gmail.com'),
    'password': os.getenv('EMAIL_PASSWORD', 'your-app-password'),
    'use_tls': True
}

# =========================================================
# Notification Settings
# =========================================================
NOTIFICATION_CONFIG = {
    'email_notifications': True,
    'attendance_threshold': 75,
    'late_threshold_minutes': 15,
    'absent_notification_delay_hours': 2
}

# =========================================================
# File Paths
# =========================================================
PHOTOS_DIR = "data/photos"
MODELS_DIR = "models"
EXPORTS_DIR = "data/exports"
LOGS_DIR = "logs"
TEMP_DIR = "data/temp"
BACKUP_DIR = "data/backups"

directories = [PHOTOS_DIR, MODELS_DIR, EXPORTS_DIR, LOGS_DIR, TEMP_DIR, BACKUP_DIR]
for directory in directories:
    os.makedirs(directory, exist_ok=True)

# =========================================================
# Academic Configuration
# =========================================================
ACADEMIC_CONFIG = {
    'academic_year_start_month': 7,
    'semester_duration_months': 6,
    'default_class_duration_minutes': 60,
    'attendance_grace_period_minutes': 10
}

# =========================================================
# Security Configuration
# =========================================================
SECURITY_CONFIG = {
    'encrypt_face_data': True,
    'secure_file_storage': True,
    'audit_logs': True,
    'max_file_uploads_per_hour': 100,
    'allowed_image_types': ['jpg', 'jpeg', 'png', 'bmp'],
    'scan_uploads_for_malware': False
}

# =========================================================
# System Monitoring
# =========================================================
MONITORING_CONFIG = {
    'enable_performance_monitoring': True,
    'cpu_threshold': 80,
    'memory_threshold': 85,
    'disk_threshold': 90,
    'monitor_interval_seconds': 60
}

# =========================================================
# UI Configuration
# =========================================================
UI_CONFIG = {
    'theme': 'light',
    'primary_color': '#1f77b4',
    'sidebar_expanded': True,
    'show_tips': True,
    'animation_enabled': True,
    'charts_theme': 'plotly'
}

# =========================================================
# Default Admin User
# =========================================================
DEFAULT_ADMIN = {
    'username': 'admin',
    'password': 'admin123',
    'email': 'admin@school.edu',
    'role': 'admin'
}
