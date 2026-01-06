"""Widen reasoning_traces.id

Revision ID: 2a4c6e9f7b10
Revises: 9a8b5c1d2e3f
Create Date: 2026-01-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "2a4c6e9f7b10"
down_revision: Union[str, None] = "9a8b5c1d2e3f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "reasoning_traces",
        "id",
        existing_type=sa.String(length=36),
        type_=sa.String(length=100),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "reasoning_traces",
        "id",
        existing_type=sa.String(length=100),
        type_=sa.String(length=36),
        existing_nullable=False,
    )
