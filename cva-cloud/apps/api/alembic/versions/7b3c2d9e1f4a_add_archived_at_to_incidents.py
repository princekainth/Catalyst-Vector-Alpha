"""add archived_at to incidents

Revision ID: 7b3c2d9e1f4a
Revises: 2a4c6e9f7b10
Create Date: 2026-01-06

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "7b3c2d9e1f4a"
down_revision: Union[str, None] = "2a4c6e9f7b10"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "incidents",
        sa.Column("archived_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("incidents", "archived_at")
