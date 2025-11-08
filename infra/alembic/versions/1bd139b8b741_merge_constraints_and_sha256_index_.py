"""merge constraints and sha256 index branches

Revision ID: 1bd139b8b741
Revises: b3053276e058, d6a8a1c4a1a9
Create Date: 2025-11-07 20:39:07.833283

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1bd139b8b741'
down_revision: Union[str, Sequence[str], None] = ('b3053276e058', 'd6a8a1c4a1a9')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
