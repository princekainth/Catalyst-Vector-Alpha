"""add pod_snapshot to clusters

Revision ID: 8b85164d4921
Revises: 9df10f6b97f2
Create Date: 2026-01-04

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "8b85164d4921"
down_revision: Union[str, None] = "9df10f6b97f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    column_type = (
        sa.JSON()
        if bind.dialect.name == "sqlite"
        else postgresql.JSONB(astext_type=sa.Text())
    )
    op.add_column(
        "clusters",
        sa.Column("pod_snapshot", column_type, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("clusters", "pod_snapshot")
