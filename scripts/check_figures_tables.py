#!/usr/bin/env python3
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import sessionmaker
from src.app.db.models.models import Figure, TableSet
from uuid import UUID

async def check_data():
    # Create engine using async URL
    engine = create_async_engine("postgresql+asyncpg://kl:klpass@localhost:5432/knowledge")
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    container_id = UUID("9e72ca18-4ecc-4178-bec5-7c0396899d0f")
    
    async with async_session() as session:
        # Count figures
        figure_count = await session.execute(
            select(func.count()).select_from(Figure).where(Figure.container_id == container_id)
        )
        figures = figure_count.scalar()
        
        # Count tables
        table_count = await session.execute(
            select(func.count()).select_from(TableSet).where(TableSet.container_id == container_id)
        )
        tables = table_count.scalar()
        
        # Get some figure details
        figure_results = await session.execute(
            select(Figure).where(Figure.container_id == container_id).limit(5)
        )
        figure_list = figure_results.scalars().all()
        
        # Get some table details
        table_results = await session.execute(
            select(TableSet).where(TableSet.container_id == container_id).limit(5)
        )
        table_list = table_results.scalars().all()
        
    await engine.dispose()
    
    print(f"Container: {container_id}")
    print(f"Figures in DB: {figures}")
    print(f"Tables in DB: {tables}")
    
    print("\nFirst few figures:")
    for fig in figure_list:
        print(f"  - Figure {fig.figure_id} on page {fig.page_no}, URI: {fig.image_uri}")
    
    print("\nFirst few tables:")
    for tbl in table_list:
        print(f"  - Table '{tbl.name}' on page {tbl.page_no}")

if __name__ == "__main__":
    asyncio.run(check_data())
