"""Add missing cluster and incident columns

Revision ID: 4c7f5a0f3c2a
Revises: 1b0b20b7c9b1
Create Date: 2026-01-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "4c7f5a0f3c2a"
down_revision: Union[str, None] = "1b0b20b7c9b1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # clusters: user_id, api_key, created_at
    op.add_column(
        "clusters",
        sa.Column("user_id", sa.String(length=100), nullable=False, server_default="demo-org"),
    )
    op.add_column(
        "clusters",
        sa.Column("api_key", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "clusters",
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )

    # incidents: user_id + workflow fields
    op.add_column(
        "incidents",
        sa.Column("user_id", sa.String(length=100), nullable=False, server_default="demo-org"),
    )
    op.add_column(
        "incidents",
        sa.Column("namespace", sa.String(length=255), nullable=True, server_default="default"),
    )
    op.add_column(
        "incidents",
        sa.Column("action_type", sa.String(length=100), nullable=True, server_default=""),
    )
    op.add_column(
        "incidents",
        sa.Column("action_config", sa.Text(), nullable=True, server_default=""),
    )
    op.add_column(
        "incidents",
        sa.Column("outcome", sa.Text(), nullable=True, server_default=""),
    )
    op.add_column(
        "incidents",
        sa.Column("executed_at", sa.DateTime(), nullable=True),
    )
    op.add_column(
        "incidents",
        sa.Column("completed_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("incidents", "completed_at")
    op.drop_column("incidents", "executed_at")
    op.drop_column("incidents", "outcome")
    op.drop_column("incidents", "action_config")
    op.drop_column("incidents", "action_type")
    op.drop_column("incidents", "namespace")
    op.drop_column("incidents", "user_id")

    op.drop_column("clusters", "created_at")
    op.drop_column("clusters", "api_key")
    op.drop_column("clusters", "user_id")
