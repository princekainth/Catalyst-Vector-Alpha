from datetime import datetime

from sqlalchemy import String, DateTime, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Incident(Base):
    __tablename__ = "incidents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    cluster_id: Mapped[str] = mapped_column(String(36), ForeignKey("clusters.id"))
    user_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    namespace: Mapped[str] = mapped_column(String(255), default="default")
    pod_name: Mapped[str] = mapped_column(String(255))
    issue_type: Mapped[str] = mapped_column(String(100))
    severity: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(50), default="pending")
    summary: Mapped[str] = mapped_column(Text, default="")
    action_type: Mapped[str] = mapped_column(String(100), default="", nullable=True)
    action_config: Mapped[str] = mapped_column(Text, default="", nullable=True)
    outcome: Mapped[str] = mapped_column(Text, default="", nullable=True)
    executed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    cluster = relationship("Cluster", back_populates="incidents")
    reasoning_traces = relationship("ReasoningTrace", back_populates="incident")
    actions = relationship("Action", back_populates="incident")
