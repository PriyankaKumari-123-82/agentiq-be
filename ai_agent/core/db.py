import sqlmodel
from sqlmodel import Session, create_engine
from typing import Generator

from ai_agent.core.config import settings

engine = create_engine(str(settings.DATABASE_URL), pool_size=settings.DATABASE_POOL_SIZE)


# make sure all SQLModel models are imported (app.models) before initializing DB
# otherwise, SQLModel might fail to initialize relationships properly
# for more details: https://github.com/fastapi/full-stack-fastapi-template/issues/28


def init_db(session: Session) -> None:
    # Tables should be created with Alembic migrations
    # But if you don't want to use migrations, create
    # the tables un-commenting the next lines
    # from sqlmodel import SQLModel

    # This works because the models are already imported and registered from app.models
    # SQLModel.metadata.create_all(engine)
    pass


def get_session() -> Generator[sqlmodel.Session, None, None]:
    session = sqlmodel.Session(engine)
    try:
        yield session
    finally:
        session.close()
