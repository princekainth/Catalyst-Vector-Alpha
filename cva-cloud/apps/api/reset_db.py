from app.db.base import Base
from app.db.session import engine
from app import models  # noqa: F401


def reset_database() -> None:
    print("Resetting database: dropping all tables...")
    Base.metadata.drop_all(bind=engine)
    print("Creating all tables from models...")
    Base.metadata.create_all(bind=engine)
    print("Database reset complete.")


if __name__ == "__main__":
    reset_database()
