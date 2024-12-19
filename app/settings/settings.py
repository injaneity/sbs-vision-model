from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    roboflow_api_key: str

    class Config:
        env_file = ".env"

settings = Settings()
