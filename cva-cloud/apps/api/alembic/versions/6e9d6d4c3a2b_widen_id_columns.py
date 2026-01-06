"""Widen id columns for org/cluster/user references

Revision ID: 6e9d6d4c3a2b
Revises: 4c7f5a0f3c2a
Create Date: 2026-01-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "6e9d6d4c3a2b"
down_revision: Union[str, None] = "4c7f5a0f3c2a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "organizations",
        "id",
        existing_type=sa.String(length=36),
        type_=sa.String(length=100),
        existing_nullable=False,
    )
    op.alter_column(
        "users",
        "id",
        existing_type=sa.String(length=36),
        type_=sa.String(length=100),
        existing_nullable=False,
    )
    op.alter_column(
        "users",
        "org_id",
        existing_type=sa.String(length=36),
        type_=sa.String(length=100),
        existing_nullable=False,
    )
    op.alter_column(
        "clusters",
        "id",
        existing_type=sa.String(length=36),
        type_=sa.String(length=100),
        existing_nullable=False,
    )
    op.alter_column(
        "clusters",
        "org_id",
        existing_type=sa.String(length=36),
        type_=sa.String(length=100),
        existing_nullable=False,
    )
    op.alter_column(
        "incidents",
        "cluster_id",
        existing_type=sa.String(length=36),
        type_=sa.String(length=100),
        existing_nullable=False,
    )
    op.alter_column(
        "actions",
        "incident_id",
        existing_type=sa.String(length=36),
        type_=sa.String(length=100),
        existing_nullable=False,
    )
    op.alter_column(
        "reasoning_traces",
        "incident_id",
        existing_type=sa.String(length=36),
        type_=sa.String(length=100),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "reasoning_traces",
        "incident_id",
        existing_type=sa.String(length=100),
        type_=sa.String(length=36),
        existing_nullable=False,
    )
    op.alter_column(
        "actions",
        "incident_id",
        existing_type=sa.String(length=100),
        type_=sa.String(length=36),
        existing_nullable=False,
    )
    op.alter_column(
        "incidents",
        "cluster_id",
        existing_type=sa.String(length=100),
        type_=sa.String(length=36),
        existing_nullable=False,
    )
    op.alter_column(
        "clusters",
        "org_id",
        existing_type=sa.String(length=100),
        type_=sa.String(length=36),
        existing_nullable=False,
    )
    op.alter_column(
        "clusters",
        "id",
        existing_type=sa.String(length=100),
        type_=sa.String(length=36),
        existing_nullable=False,
    )
    op.alter_column(
        "users",
        "org_id",
        existing_type=sa.String(length=100),
        type_=sa.String(length=36),
        existing_nullable=False,
    )
    op.alter_column(
        "users",
        "id",
        existing_type=sa.String(length=100),
        type_=sa.String(length=36),
        existing_nullable=False,
    )
    op.alter_column(
        "organizations",
        "id",
        existing_type=sa.String(length=100),
        type_=sa.String(length=36),
        existing_nullable=False,
    )
