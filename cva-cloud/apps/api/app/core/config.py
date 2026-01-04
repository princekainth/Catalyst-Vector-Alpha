from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "CVA Cloud API"
    env: str = "development"
    api_v1_prefix: str = "/api/v1"

    database_url: str = "sqlite:///./cva.db"
    redis_url: str = "redis://localhost:6379/0"

    clerk_jwks_url: str = ""
    clerk_issuer: str = ""
    clerk_audience: str = ""

    class Config:
        env_prefix = "CVA_"
        case_sensitive = False


settings = Settings()
