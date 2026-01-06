"""Add policies table

Revision ID: 1b0b20b7c9b1
Revises: 8b85164d4921
Create Date: 2026-01-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "1b0b20b7c9b1"
down_revision: Union[str, None] = "8b85164d4921"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "policies",
        sa.Column("id", sa.String(length=100), nullable=False),
        sa.Column("org_id", sa.String(length=100), nullable=False),
        sa.Column("cluster_id", sa.String(length=100), nullable=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("issue_type", sa.String(length=100), nullable=False),
        sa.Column("auto_approve", sa.Boolean(), nullable=False),
        sa.Column("max_memory_mb", sa.Integer(), nullable=True),
        sa.Column("allow_placeholder", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["org_id"], ["organizations.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_policies_org_id", "policies", ["org_id"])


def downgrade() -> None:
    op.drop_index("ix_policies_org_id", table_name="policies")
    op.drop_table("policies")
