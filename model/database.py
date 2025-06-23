import os
from datetime import datetime

import pytz
from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Get database URL from environment variable or use default
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://alexanderdavidson@localhost:5432/mnist_predictions"
)

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


def get_local_time():
    return datetime.utcnow()


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), default=get_local_time)
    predicted_digit = Column(Integer)
    true_label = Column(
        Integer, nullable=True
    )  # Nullable because we might not always have the true label


# Create tables
Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
