#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import logging
from typing import Optional
from uuid import UUID

from src.app.db.session.session_async import AsyncSessionLocal
from src.app.services.ingest.container_pipeline import (
    ingest_pdf_container_pipeline,
    process_pdf_container_async,
)
from src.app.api.v1.ingest.schemas import IngestOptions


async def _run(pdf_path: Path, collection_id: Optional[UUID]) -> None:
    # Create container record and then process in the same run (no API server needed)
    async with AsyncSessionLocal() as session:
        resp = await ingest_pdf_container_pipeline(
            session,
            pdf_path=pdf_path,
            options=IngestOptions(dedupe=False),
            source_uri=f"file://{pdf_path}",
            title_hint=pdf_path.stem,
            collection_id=collection_id,
        )
        container_id = resp.container_id
    print(f"Created container: {container_id}")
    print("Processing PDF pages (layout + text/ocr + figures/tables)...")
    await process_pdf_container_async(container_id=container_id, pdf_path=pdf_path)
    print("Done.")


def main() -> int:
    p = argparse.ArgumentParser(description="Process a local PDF through the container pipeline without starting the API")
    p.add_argument("path", help="Path to PDF file")
    p.add_argument("--collection-id", default=None, help="Optional collection UUID to associate")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"], help="Logging level (default: %(default)s)")
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    pdf = Path(args.path).expanduser().resolve()
    if not pdf.is_file():
        print(f"Not a file: {pdf}")
        return 2

    coll: Optional[UUID] = None
    if args.collection_id:
        try:
            coll = UUID(args.collection_id)
        except Exception:
            print("Invalid collection_id; ignoring.")

    asyncio.run(_run(pdf, coll))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
