"""PDF parser plugin using Marker library.

Extracts text, heading hierarchy, tables, and image positions from PDF files.
Handles corrupted and password-protected PDFs gracefully.

环境变量 ``PDF_PARSER_MODE`` 控制解析后端:
- ``marker`` (默认): 使用 Marker (layout 感知 + OCR), 模型权重 ~3GB,
  内存峰值 3-4GB, 适合公式 / 复杂排版的扫描件。
- ``fitz``: 使用 PyMuPDF, 速度快, 内存 <100MB, 不识别图像但保留文本和
  字号 (heading 检测靠字号 + bold)。低内存环境 (<8GB Docker VM) 推荐。
- ``auto``: 文件 <2MB 用 fitz, 否则用 marker (兼顾速度和精度)。
"""

import logging
import os
import uuid

from app.services.parsers.base import Asset, Block, ParsedDocument, ParseError

logger = logging.getLogger(__name__)


# Auto 模式的文件大小阈值 (字节). 默认 2MB:
# - 简历 / 短文档 (<2MB): fitz 几秒搞定
# - 大文档 / 扫描件: 走 marker 拿到 OCR + layout
_AUTO_FITZ_MAX_BYTES = int(os.environ.get("PDF_PARSER_AUTO_FITZ_MAX_BYTES", 2 * 1024 * 1024))


class PdfParser:
    """PDF parser plugin using Marker for high-quality extraction.

    Marker provides layout-aware PDF parsing with support for:
    - Text extraction with position information
    - Heading level detection
    - Table structure recognition
    - Image extraction with bounding boxes
    """

    name: str = "pdf-parser"
    supported_extensions: list[str] = ["pdf"]
    priority: int = 100

    def can_parse(self, file_path: str, mime_type: str) -> bool:
        """Check if this parser can handle the file."""
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        return ext in self.supported_extensions or mime_type == "application/pdf"

    async def parse(self, file_path: str) -> ParsedDocument:
        """Parse a PDF file using Marker.

        Args:
            file_path: Path to the PDF file

        Returns:
            ParsedDocument with extracted blocks, metadata, and assets

        Raises:
            ParseError: If the PDF is corrupted or password-protected
        """
        if not os.path.exists(file_path):
            raise ParseError(f"File not found: {file_path}", reason="corrupted")

        mode = (os.environ.get("PDF_PARSER_MODE") or "marker").strip().lower()
        if mode == "auto":
            try:
                size = os.path.getsize(file_path)
            except OSError:
                size = 0
            mode = "fitz" if 0 < size <= _AUTO_FITZ_MAX_BYTES else "marker"

        logger.info("PDF parser mode=%s for %s", mode, file_path)

        try:
            if mode == "fitz":
                return await self._parse_basic(file_path)
            return await self._parse_with_marker(file_path)
        except ParseError:
            raise
        except Exception as e:
            error_msg = str(e).lower()
            if "password" in error_msg or "encrypted" in error_msg:
                raise ParseError(
                    f"PDF is password-protected: {file_path}",
                    reason="password_protected",
                )
            if "corrupt" in error_msg or "invalid" in error_msg or "damaged" in error_msg:
                raise ParseError(
                    f"PDF file is corrupted: {file_path}",
                    reason="corrupted",
                )
            raise ParseError(
                f"Failed to parse PDF: {file_path}: {e}",
                reason="unknown",
            )

    async def _parse_with_marker(self, file_path: str) -> ParsedDocument:
        """Internal method to parse PDF using Marker library."""
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
        except ImportError:
            # Fallback to basic extraction if marker is not available
            return await self._parse_basic(file_path)

        try:
            converter = PdfConverter(artifact_dict=create_model_dict())
            rendered = converter(file_path)
        except Exception as e:
            error_msg = str(e).lower()
            if "password" in error_msg or "encrypted" in error_msg:
                raise ParseError(
                    f"PDF is password-protected: {file_path}",
                    reason="password_protected",
                )
            raise

        blocks: list[Block] = []
        assets: list[Asset] = []
        metadata: dict = {"source": file_path}

        # Process rendered output from Marker
        if hasattr(rendered, "children"):
            page_num = 1
            for page in rendered.children:
                page_num = getattr(page, "page_id", page_num)
                for block in getattr(page, "children", []):
                    parsed_block = self._convert_marker_block(block, page_num)
                    if parsed_block:
                        blocks.append(parsed_block)
                page_num += 1
        elif hasattr(rendered, "markdown"):
            # Marker v1 returns markdown directly
            blocks = self._parse_markdown_output(rendered.markdown)
            metadata["page_count"] = getattr(rendered, "page_count", 1)

        if not blocks:
            raise ParseError(
                f"No content extracted from PDF: {file_path}",
                reason="empty",
            )

        return ParsedDocument(blocks=blocks, metadata=metadata, assets=assets)

    async def _parse_basic(self, file_path: str) -> ParsedDocument:
        """Basic PDF text extraction fallback using PyPDF2 or pdfplumber."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ParseError(
                "No PDF parsing library available (marker or PyMuPDF required)",
                reason="unknown",
            )

        blocks: list[Block] = []
        assets: list[Asset] = []
        metadata: dict = {"source": file_path}

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            if "password" in str(e).lower() or "encrypted" in str(e).lower():
                raise ParseError(
                    f"PDF is password-protected: {file_path}",
                    reason="password_protected",
                )
            raise ParseError(
                f"Cannot open PDF: {file_path}: {e}",
                reason="corrupted",
            )

        metadata["page_count"] = len(doc)

        for page_num, page in enumerate(doc, start=1):
            # Extract text blocks
            text_dict = page.get_text("dict")
            for block_data in text_dict.get("blocks", []):
                if block_data.get("type") == 0:  # Text block
                    text = ""
                    for line in block_data.get("lines", []):
                        for span in line.get("spans", []):
                            text += span.get("text", "")
                        text += "\n"
                    text = text.strip()
                    if text:
                        bbox_raw = block_data.get("bbox", (0, 0, 0, 0))
                        page_rect = page.rect
                        # Normalize bbox to 0-1 range
                        bbox = (
                            bbox_raw[0] / page_rect.width if page_rect.width else 0,
                            bbox_raw[1] / page_rect.height if page_rect.height else 0,
                            bbox_raw[2] / page_rect.width if page_rect.width else 0,
                            bbox_raw[3] / page_rect.height if page_rect.height else 0,
                        )
                        # Detect heading by font size
                        font_size = 0
                        is_bold = False
                        for line in block_data.get("lines", []):
                            for span in line.get("spans", []):
                                font_size = max(font_size, span.get("size", 0))
                                if "bold" in span.get("font", "").lower():
                                    is_bold = True

                        block_type = "paragraph"
                        style: dict = {"font_size": font_size, "bold": is_bold}
                        if font_size >= 16 or is_bold:
                            block_type = "heading"
                            if font_size >= 24:
                                style["heading_level"] = 1
                            elif font_size >= 20:
                                style["heading_level"] = 2
                            elif font_size >= 16:
                                style["heading_level"] = 3
                            else:
                                style["heading_level"] = 4

                        blocks.append(Block(
                            type=block_type,
                            text=text,
                            bbox=bbox,
                            page_number=page_num,
                            style=style,
                            raw={"block_type": block_data.get("type")},
                        ))

                elif block_data.get("type") == 1:  # Image block
                    bbox_raw = block_data.get("bbox", (0, 0, 0, 0))
                    page_rect = page.rect
                    bbox = (
                        bbox_raw[0] / page_rect.width if page_rect.width else 0,
                        bbox_raw[1] / page_rect.height if page_rect.height else 0,
                        bbox_raw[2] / page_rect.width if page_rect.width else 0,
                        bbox_raw[3] / page_rect.height if page_rect.height else 0,
                    )
                    # Extract image data
                    try:
                        image_data = block_data.get("image", b"")
                        if image_data:
                            asset_id = str(uuid.uuid4())
                            assets.append(Asset(
                                id=asset_id,
                                type="image",
                                data=image_data if isinstance(image_data, bytes) else b"",
                                mime_type="image/png",
                                page_number=page_num,
                                bbox=bbox,
                            ))
                            blocks.append(Block(
                                type="image",
                                text=f"[Image: {asset_id}]",
                                bbox=bbox,
                                page_number=page_num,
                                style={},
                                raw={"asset_id": asset_id},
                            ))
                    except Exception:
                        pass

        doc.close()

        if not blocks:
            raise ParseError(
                f"No content extracted from PDF: {file_path}",
                reason="empty",
            )

        return ParsedDocument(blocks=blocks, metadata=metadata, assets=assets)

    def _convert_marker_block(self, block: object, page_num: int) -> Block | None:
        """Convert a Marker block object to our Block dataclass."""
        block_type = getattr(block, "block_type", "paragraph")
        text = getattr(block, "text", "") or getattr(block, "content", "")

        if not text and block_type != "image":
            return None

        style: dict = {}
        if block_type == "heading" or block_type.startswith("heading"):
            level = getattr(block, "level", 1)
            style["heading_level"] = level
            block_type = "heading"
        elif block_type == "table":
            block_type = "table"
        elif block_type == "image":
            block_type = "image"
        else:
            block_type = "paragraph"

        bbox = getattr(block, "bbox", None)
        if bbox and len(bbox) == 4:
            bbox = tuple(bbox)
        else:
            bbox = None

        return Block(
            type=block_type,
            text=text.strip() if text else "",
            bbox=bbox,
            page_number=page_num,
            style=style,
            raw={},
        )

    def _parse_markdown_output(self, markdown: str) -> list[Block]:
        """Parse Marker's markdown output into blocks."""
        blocks: list[Block] = []
        lines = markdown.split("\n")
        current_text = ""
        current_type = "paragraph"
        current_style: dict = {}

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current_text:
                    blocks.append(Block(
                        type=current_type,
                        text=current_text.strip(),
                        page_number=1,
                        style=current_style,
                    ))
                    current_text = ""
                    current_type = "paragraph"
                    current_style = {}
                continue

            # Detect headings
            if stripped.startswith("#"):
                if current_text:
                    blocks.append(Block(
                        type=current_type,
                        text=current_text.strip(),
                        page_number=1,
                        style=current_style,
                    ))
                    current_text = ""

                level = 0
                for ch in stripped:
                    if ch == "#":
                        level += 1
                    else:
                        break
                heading_text = stripped[level:].strip()
                blocks.append(Block(
                    type="heading",
                    text=heading_text,
                    page_number=1,
                    style={"heading_level": min(level, 6)},
                ))
                current_type = "paragraph"
                current_style = {}
            elif stripped.startswith("|") and "|" in stripped[1:]:
                # Table row
                if current_text:
                    blocks.append(Block(
                        type=current_type,
                        text=current_text.strip(),
                        page_number=1,
                        style=current_style,
                    ))
                    current_text = ""
                current_type = "table"
                current_text += line + "\n"
                current_style = {}
            else:
                if current_type == "table" and not stripped.startswith("|"):
                    blocks.append(Block(
                        type="table",
                        text=current_text.strip(),
                        page_number=1,
                        style=current_style,
                    ))
                    current_text = ""
                    current_type = "paragraph"
                    current_style = {}
                current_text += line + "\n"

        if current_text:
            blocks.append(Block(
                type=current_type,
                text=current_text.strip(),
                page_number=1,
                style=current_style,
            ))

        return blocks
