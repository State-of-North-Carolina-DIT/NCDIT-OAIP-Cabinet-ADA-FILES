"""
Regenerate sample_pages_review.html from existing JSON output files.
This avoids rerunning the full extraction pipeline.
"""

import json
import random
import base64
from pathlib import Path

import pypdfium2 as pdfium

from extract_structured_json import (
    render_content_item_html, CONTENT_CSS,
    DATA_FOLDER, OUTPUT_FOLDER, REPORTS_FOLDER, RENDER_SCALE,
)

# Image scale: reuse RENDER_SCALE from the main script (env-var-backed)
PAGE_IMAGE_SCALE = float(RENDER_SCALE)


def render_page_to_image(pdf_path: Path, page_index: int) -> bytes:
    """Render a PDF page to PNG bytes."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf[page_index]
    bitmap = page.render(scale=PAGE_IMAGE_SCALE)
    pil_image = bitmap.to_pil()

    import io
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def generate_sample_pages_report(results: list, num_samples: int = 30):
    """
    Generate an HTML report with sample pages for human spot-checking.

    Selects a mix of:
    - Lowest confidence pages (for review)
    - Random pages (for unbiased sampling)
    - Pages with coherence issues
    """
    # Collect all pages with their metadata
    all_pages = []
    for result in results:
        pdf_id = result["pdf_id"]
        pdf_path = Path(result.get("source_path", DATA_FOLDER / pdf_id / "source.pdf"))

        if not pdf_path.exists():
            print(f"Warning: PDF not found: {pdf_path}")
            continue

        for page in result.get("pages", []):
            if page.get("error"):
                continue

            val = page.get("validation", {})
            coherence = val.get("coherence_score")
            if coherence is None:
                coherence = 8  # Default

            all_pages.append({
                "pdf_id": pdf_id,
                "pdf_path": pdf_path,
                "page_number": page["page_number"],
                "content": page.get("content", []),
                "coherence_score": coherence,
                "coherence_issues": val.get("coherence_issues", []),
            })

    if not all_pages:
        print("No pages found to include in report.")
        return

    # Select samples
    samples = []

    # 1. Lowest coherence score pages (10)
    sorted_by_score = sorted(all_pages, key=lambda x: x["coherence_score"])
    samples.extend(sorted_by_score[:10])

    # 2. Pages with coherence issues (10)
    pages_with_issues = [p for p in all_pages if p["coherence_issues"]]
    random.shuffle(pages_with_issues)
    for page in pages_with_issues[:10]:
        if page not in samples:
            samples.append(page)

    # 3. Random pages to fill up to num_samples
    remaining = [p for p in all_pages if p not in samples]
    random.shuffle(remaining)
    for page in remaining:
        if len(samples) >= num_samples:
            break
        samples.append(page)

    # Sort samples by coherence score for review priority
    samples.sort(key=lambda x: x["coherence_score"])

    # Generate HTML report
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Sample Pages for Human Review</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .page-sample {
            border: 1px solid #ddd;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            background: #fafafa;
        }
        .page-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #ddd;
        }
        .score-badge {
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: bold;
        }
        .score-high { background: #c8e6c9; color: #2e7d32; }
        .score-medium { background: #fff3cd; color: #856404; }
        .score-low { background: #ffcdd2; color: #c62828; }
        .content-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .original-image {
            max-width: 100%;
            border: 1px solid #ccc;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .extracted-content {
            background: white;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
            max-height: 800px;
            overflow-y: auto;
        }
        .content-item {
            margin-bottom: 10px;
            padding: 8px;
            border-left: 3px solid #2196F3;
            background: #f5f5f5;
        }
        .content-type {
            font-weight: bold;
            color: #666;
            font-size: 0.85em;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .issues-list {
            background: #fff3cd;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
        }
        .issues-list li { color: #856404; }
        table.extracted-table {
            border-collapse: collapse;
            width: 100%;
            font-size: 0.9em;
        }
        table.extracted-table th, table.extracted-table td {
            border: 1px solid #ddd;
            padding: 4px 8px;
        }
        table.extracted-table th { background: #e0e0e0; }
        .note { background: #e3f2fd; padding: 15px; border-radius: 4px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <h1>Sample Pages for Human Review</h1>
    <div class="note">
        <p><strong>Instructions:</strong> Review these sample pages to verify extraction quality.
        Pages are sorted by confidence score (lowest first). Check that:</p>
        <ul>
            <li>All text from the original is captured</li>
            <li>Tables are correctly structured</li>
            <li>Reading order is preserved</li>
            <li>No garbled or missing text</li>
        </ul>
    </div>
"""

    for i, sample in enumerate(samples):
        # Render page image
        try:
            page_image_bytes = render_page_to_image(
                sample["pdf_path"], sample["page_number"] - 1
            )
            image_base64 = base64.b64encode(page_image_bytes).decode("utf-8")
            image_html = f'<img class="original-image" src="data:image/png;base64,{image_base64}" alt="Page {sample["page_number"]}">'
        except Exception as e:
            image_html = f'<p style="color: red;">Failed to render image: {e}</p>'

        # Determine score class based on coherence score
        if sample["coherence_score"] >= 9:
            score_class = "score-high"
        elif sample["coherence_score"] >= 7:
            score_class = "score-medium"
        else:
            score_class = "score-low"

        # Format extracted content using shared renderer
        content_html = ""
        for item in sample["content"]:
            content_html += render_content_item_html(item)

        # Format issues
        issues_html = ""
        if sample["coherence_issues"]:
            issues_html = '<div class="issues-list"><strong>Coherence Issues:</strong><ul>'
            for issue in sample["coherence_issues"]:
                issues_html += f"<li>{issue}</li>"
            issues_html += "</ul></div>"

        html += f'''
    <div class="page-sample">
        <div class="page-header">
            <h2>#{i+1}: {sample["pdf_id"]} - Page {sample["page_number"]}</h2>
            <div>
                <span class="score-badge {score_class}">
                    Coherence: {sample["coherence_score"]}/10
                </span>
            </div>
        </div>
        {issues_html}
        <div class="content-grid">
            <div>
                <h3>Original Page</h3>
                {image_html}
            </div>
            <div>
                <h3>Extracted Content</h3>
                <div class="extracted-content">
                    {content_html if content_html else '<p style="color: #999;">No content extracted</p>'}
                </div>
            </div>
        </div>
    </div>
'''

    html += """
</body>
</html>"""

    # Save report
    REPORTS_FOLDER.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_FOLDER / "sample_pages_review.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Sample pages report saved: {report_path}")
    print(f"Total samples included: {len(samples)}")


def main():
    """Load all JSON files from output/ and regenerate the sample review report."""
    print("Loading JSON files from output/...")

    results = []
    json_files = list(OUTPUT_FOLDER.glob("*.json"))

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    print(f"Loaded {len(results)} JSON files")

    if not results:
        print("No JSON files found in output/")
        return

    generate_sample_pages_report(results)


if __name__ == "__main__":
    main()
