"""Make clusters.last_seen and agent_version nullable

Revision ID: 9a8b5c1d2e3f
Revises: 6e9d6d4c3a2b
Create Date: 2026-01-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9a8b5c1d2e3f"
down_revision: Union[str, None] = "6e9d6d4c3a2b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "clusters",
        "last_seen",
        existing_type=sa.DateTime(),
        nullable=True,
    )
    op.alter_column(
        "clusters",
        "agent_version",
        existing_type=sa.String(length=50),
        nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "clusters",
        "agent_version",
        existing_type=sa.String(length=50),
        nullable=False,
    )
    op.alter_column(
        "clusters",
        "last_seen",
        existing_type=sa.DateTime(),
        nullable=False,
    )
