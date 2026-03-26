Act as an expert in parsing PDF files. Examine this image of a PDF page and extract ALL textual and visual content into a structured JSON array.

IMPORTANT: You MUST return valid JSON. Do not include any text before or after the JSON array.

## Output Format
Return a JSON array where each item has a "type" field:

### Headings
{"type": "heading", "level": <1-6>, "text": "Heading text"}
- Use for titles, section headers, and any visually prominent standalone text
- Level 1: Document title, main title on a cover page (use sparingly — typically only one per document)
- Level 2: Major section headings
- Level 3: Subsection headings
- Level 4-6: Progressively smaller sub-subsections
- Determine level by visual prominence: font size, boldness, spacing
- CRITICAL: Maintain proper heading hierarchy. If the document has a clear outline structure (e.g., sections with subsections), subsections MUST use a deeper level than their parent. Do NOT make all headings the same level (e.g., all H2).
- Text should be plain text (no markdown # markers, no ** bold markers)
- If unsure between heading and paragraph, use heading for short, prominent, standalone text
- Do NOT promote regular body text to headings just because it is bold or italic
- CRITICAL: Do NOT extract text that appears INSIDE diagram boxes, flow chart bubbles, infographic callout boxes, chart labels, or organizational chart nodes as heading objects. Text within visual/diagrammatic elements should be described in the associated image's "description" field, not as standalone headings.
- CRITICAL: Do NOT make person names, organization names, or short label text from within infographics into headings. Only use heading type for actual document section titles.
- If you see numbered items like "1. Introduction", "2. Background" that form the document outline, use the SAME heading level for items at the SAME logical depth — do NOT assign different levels to parallel outline items.

### Paragraphs
{"type": "paragraph", "text": "Markdown formatted text..."}
- Use for body text and regular content (NOT headings, NOT lists, NOT headers/footers)
- Preserve bold (**text**) and italics (*text*) ONLY when the original PDF uses bold or italic formatting
- CRITICAL: Do NOT add ** or * around text that is not bold or italic in the original PDF. Only use markdown formatting when the visual appearance clearly shows bold or italic styling
- Follow natural reading order (see Multi-Column Layouts section below)
- Do NOT put bulleted or numbered lists inside paragraphs - use the "list" type instead
- Do NOT put page headers/footers in paragraphs - use the "header_footer" type instead

### Tables
{"type": "table", "cells": [...]}
Each cell object must include:
- text: Plain text content of the cell (no markdown formatting — do NOT wrap cell text in ** or *)
- column_start: 0-indexed column position
- row_start: 0-indexed row position
- num_columns: columns spanned (default 1)
- num_rows: rows spanned (default 1)

IMPORTANT for tables:
- CRITICAL: Do NOT add ** asterisks or * to table cell text. Table header cells are identified by position, not by markdown bold. Write cell text as plain text only.
- Count grid lines to determine exact rows/columns
- Handle merged cells by looking at grid boundaries
- Table titles (e.g., "Table 2.1") should be paragraphs, not table cells
- CRITICAL: Do NOT split a single table row into multiple rows. If a cell contains text that wraps across multiple visual lines due to word wrap or small column width, it is STILL one cell in ONE row — extract all the wrapped text as a single cell value.
- CRITICAL: The first row of a table is a header row ONLY if its cells contain COLUMN LABELS (descriptive names like "Name", "Date", "Amount"). Data values (phone numbers, addresses, actual content) in the first row are NOT headers — treat them as regular data cells.
- CRITICAL: If a row contains only a single cell spanning all columns that acts as a section label within the table, represent it as a regular data row spanning all columns (with num_columns equal to the total column count). Do NOT treat it as a table heading that breaks the table into sub-tables.
- When a table continues across a page break with a repeated header row at the top of the next page, extract only the data rows from the continuation — do NOT repeat the header row.
- CRITICAL: When a table cell contains nested/indented sub-content (for example, an agenda table where the "Topic" cell for one item contains indented sub-items like "a) Minutes", "b) NG 911 Fund" with further indented financial rows like "November Fund Balance $X", "November Disbursement -$Y"), do NOT expand those nested rows into the outer table. Keep the outer table's row structure intact and include all nested sub-content as text within the relevant cell, using newlines (\n) to separate inner rows. Example: a cell whose content is "a) Minutes\nb) NG 911 Fund\nNovember Balance $X\nDisbursement -$Y" is ONE cell in ONE row of the outer table, not multiple separate rows.

### Images/Figures
{"type": "image", "description": "...", "caption": "...", "position": "..."}
- For any images, figures, diagrams, charts, or photographs
- "description" is the alt text — a concise description (1-2 sentences) of what the image VISUALLY shows. Be specific: identify the subject (person, logo, map, chart type, screenshot subject), not generic labels like "Document image" or "image"
- CRITICAL: When a page has MULTIPLE images, ensure each image's description matches THAT specific image. Do NOT swap or combine descriptions across images.
- Include the caption if one is present (e.g., "Figure 2.1: ...")
- IMPORTANT: Text directly above, below, or overlaid on an image is its caption - include it in the "caption" field, NEVER as a separate paragraph
- Position: estimate as "top/middle/bottom-left/center/right"

IMPORTANT for screenshots and UI images:
- When you see a screenshot of a software interface, email, website, or application, describe it as an IMAGE — do NOT transcribe all the text visible in the screenshot as separate paragraphs, tables, or lists
- The screenshot's visible text should be summarized in the "description" field, not extracted as separate content elements
- Only extract text from screenshots if the text IS the primary content (e.g., a scanned text document), not if it's incidental UI text

### Image Grids and Thumbnails
When you see a grid or collection of thumbnail images (like a table of contents):
- Each image with nearby text (above, below, or overlaid) is ONE image object
- Include the associated text in the "caption" field
- Include any page numbers or badges visible on/near the image in the caption
- Output each image in reading order (left-to-right, then top-to-bottom)
- Do NOT output the text labels as separate paragraphs

### Lists
{"type": "list", "list_type": "ordered"|"unordered", "items": [...]}
- Use for any bulleted lists, numbered lists, lettered lists, or step-by-step sequences
- Do NOT embed lists inside paragraphs as markdown bullets - always use a separate "list" object
- list_type: "ordered" for numbered (1, 2, 3), lettered (a, b, c), or Roman numeral (i, ii, iii) lists; "unordered" for bullet points, dashes, or other non-sequential markers

Each item in the "items" array should include:
- text: The text content of the list item (with markdown formatting preserved)
- children: Optional array of sub-items (for nested lists). Each child has a "text" field.

IMPORTANT for lists:
- Preserve the original list nesting: if an item has sub-items (indented bullets or sub-numbers like "a)", "i)"), put them in the "children" array
- Maintain reading order of list items
- If a list continues across a column break, keep it as one list object
- Agenda items with sub-items (e.g., "1. Chair's Remarks" with "a) Opening, b) Status") should use nested children
- Do NOT confuse lists with tables - if items are arranged in a grid with columns, use a table
- Short single-item bullet points that are clearly list items should still use the "list" type, not "paragraph"
- CRITICAL: If you see a list where items are labeled with letters (a., b., c., ...) or roman numerals (i., ii., iii., ...) or numbers (1., 2., 3., ...), mark the list as list_type: "ordered" — even if the visual markers look like bullets. Ordered list items must INCLUDE the letter/number prefix in the item text (e.g., "a. First item", "b. Second item") so the rendering engine can detect the list style.
- CRITICAL: Do NOT restart list numbering arbitrarily. If a numbered list continues from a previous section (e.g., a previous page ended at item 5 and this page starts with item 6), continue the sequence — do NOT restart at 1.
- When items clearly form an ordered sequence by their numbering (a, b, c or 1, 2, 3), they are one list object even if separated by paragraphs, UNLESS the paragraph clearly represents a new section break.

Example:
{"type": "list", "list_type": "ordered", "items": [
  {"text": "Chair's Opening Remarks", "children": [
    {"text": "General Opening Comments"},
    {"text": "Board Member Status"}
  ]},
  {"text": "Ethics Awareness Statement"},
  {"text": "Public Comment"}
]}

### Videos
{"type": "video", "url": "...", "description": "..."}
- ONLY use this for embedded video players visible on the page (e.g., a video thumbnail with a play button)
- Do NOT create video objects for text that contains URLs (even YouTube/Vimeo links)
- Text containing video URLs should be extracted as regular paragraphs, not video objects
- If you see "https://youtube.com/..." as printed text, that is a paragraph, NOT a video

### Forms
{"type": "form", "title": "...", "fields": [...]}
- Use for fillable forms, questionnaires, applications, checklists with input fields
- Identified by: labeled input fields, checkboxes, radio buttons, dropdown menus, text boxes with lines/boxes for input, signature lines

Each field in the "fields" array should include:
- label: The text label or question associated with the field
- field_type: One of "text", "textarea", "checkbox", "radio", "dropdown", "date", "signature", "number", "email", "phone", "unknown"
- value: The filled-in value if present. For checkboxes: "true" if checked, "false" if unchecked, null if unclear. For empty fields: null
- options: Array of visible choices (only for radio buttons and dropdowns)
- required: true if marked as required (asterisk, "required" label, etc.)
- position: Estimate location as "top/middle/bottom-left/center/right"

Field Type Detection:
- text: Single-line input with underline, box, or blank space after a label
- textarea: Multi-line text area, larger box, or "Comments/Notes" sections
- checkbox: Square box, often with checkmark or X when filled
- radio: Circle options where only one can be selected
- dropdown: Fields showing a selection arrow or "Select one"
- date: Fields labeled "Date", "DOB", or showing date format
- signature: Fields labeled "Signature" or with a signature line
- number: Fields for numeric input (amounts, quantities)
- email: Fields labeled "Email" or showing @ symbol
- phone: Fields labeled "Phone" or showing phone format

IMPORTANT for forms:
- Extract ALL visible form fields, even if empty
- For checkboxes: look for check marks, X marks, or filled boxes to determine if checked
- For handwritten entries: transcribe the handwritten text as the value
- Group related fields under a single form object when they are visually part of the same form
- Do NOT confuse data tables with forms - tables display data, forms collect input

### Headers and Footers
{"type": "header_footer", "subtype": "header"|"footer", "text": "..."}
- Use for repeating text at the top (header) or bottom (footer) of a page
- Headers: organization names, document titles, section names repeated at the top of every page
- Footers: page numbers, dates, version numbers, confidentiality notices, copyright text at the bottom
- subtype: "header" for top-of-page content, "footer" for bottom-of-page content
- Extract the FULL text including page numbers (e.g., "Page 3 of 12"), dates, and version info

IMPORTANT for headers/footers:
- Do NOT skip headers and footers - they contain provenance information (dates, versions, confidentiality)
- Do NOT put header/footer text inside paragraph objects
- If a page has both a header AND a footer, create two separate header_footer objects
- Common header patterns: organization logo text, document title, section name
- Common footer patterns: "Page X of Y", "Confidential", "Draft", dates, version numbers, copyright
- If the header/footer contains only a page number (e.g., just "3"), still extract it

Example:
{"type": "header_footer", "subtype": "header", "text": "NC Department of Information Technology - Board Meeting Agenda"}
{"type": "header_footer", "subtype": "footer", "text": "Page 3 of 12 | Confidential | Version 2.1 | March 2022"}

### Links
{"type": "link", "text": "display text", "url": "https://..."}
- Use for any text on the page that is visually styled as a hyperlink (underlined, colored, clickable-looking)
- text: The visible display text shown on the page
- url: The URL the link points to (if visible or inferrable from the text)
- IMPORTANT: When text is displayed as a URL (e.g., underlined "https://example.com"), use the URL as both "text" and "url"
- When display text differs from URL (e.g., "Click here" linking to a URL), capture the display text in "text"
- If you cannot determine the URL from the visual content alone, set url to the display text
- Note: Hyperlink URLs are also extracted programmatically from the PDF structure and merged during post-processing

IMPORTANT for links:
- Do NOT extract links as plain paragraphs - if text is visually a hyperlink, use the "link" type
- Do NOT confuse underlined text with links - only use "link" for text that appears to be a clickable hyperlink
- CRITICAL: Do NOT create hyperlinks from text visible inside screenshots or images. Only extract links from actual page text.
- CRITICAL: Do NOT fabricate or guess URLs. If you can see the display text but cannot determine the actual URL, set url to the display text. Do NOT invent URLs.
- Email addresses displayed as links (e.g., "john@example.com") should use "mailto:" prefix in url
- Multiple links in the same line should each be separate link objects
- If a link's display text is different from its URL (e.g., "Click Here" → some-url), the actual URL will be merged from PDF metadata during post-processing. Just capture the display text accurately.

## Multi-Column Layouts
CRITICAL: When a page has multiple text columns:
- Read each column completely from top to bottom BEFORE moving to the next column
- Left column first (all content), then right column (all content)
- Do NOT read across columns (row by row) - this breaks sentence continuity
- A sentence that starts at the bottom of the left column continues at the top of the right column

How to detect multi-column layouts:
- Look for a vertical gap or gutter in the middle of the page
- Text blocks that don't extend across the full page width
- Parallel paragraphs at similar heights on different sides of the page

Example (WRONG - reading across):
"The quick brown fox" | "jumped over the lazy"
"dog. Meanwhile, the" | "cat sat on the mat."
Reading: "The quick brown fox jumped over the lazy dog. Meanwhile, the cat sat on the mat."

Example (CORRECT - reading down columns):
Column 1: "The quick brown fox dog. Meanwhile, the"
Column 2: "jumped over the lazy cat sat on the mat."
Reading: "The quick brown fox dog. Meanwhile, the" then "jumped over the lazy cat sat on the mat."

## Table Extraction Guidelines
For complex tables with multi-level headers:
- Identify header rows first (usually the first 1-3 rows with column labels)
- For merged header cells (spanning multiple columns), note the num_columns value
- Data rows should have consistent column counts
- If a category label spans all columns (like a section divider), include it as a row with num_columns equal to total columns

Common table patterns:
1. Vendor comparison tables: Headers like "Dell | HP | Lenovo | Microsoft" - each vendor column should have the same number of data cells
2. Pricing tables: Item | Description | Price - ensure currency symbols stay with their numbers, not as separate columns
3. Forms/schedules: If a cell appears empty in the PDF, include it as an empty string "", don't skip it

## Quality Requirements
- Extract ALL text - do not summarize or skip content
- CRITICAL: Do NOT hallucinate or invent content. Only extract text that is actually visible on the page. Do NOT add links, text, or data that does not exist in the original document.
- CRITICAL: Do NOT add ** or * markdown formatting unless the original text is visually bold or italic
- CRITICAL: For multi-column pages, read DOWN each column completely before moving right
- Do NOT group by type - interleave paragraphs, images, lists, and tables as they appear on the page
- Be precise with table cell positions
- For tables, verify that all data rows have the same number of cells as header rows
- CRITICAL: Extract bulleted and numbered lists as "list" objects, NOT as paragraphs with markdown bullets
- Preserve list nesting (sub-items go in "children" arrays)
- CRITICAL: Extract page headers and footers as "header_footer" objects - do NOT skip them
- CRITICAL: Extract hyperlinks as "link" objects with display text and URL - do NOT flatten them into paragraphs
- CRITICAL: Preserve exact numeric values from the document. Do NOT change prices, quantities, dates, or any numerical data.
- When the same image appears as a background or decoration (e.g., a full-page slide background), do NOT transcribe its content as separate text elements. Only extract the primary content that a reader would focus on.
- CRITICAL: Do NOT truncate or stop early. If the page has 20 paragraphs, extract all 20. If a section heading appears near the bottom of the page, still include it. Extract EVERYTHING visible on the page.
- CRITICAL: Asterisks (*) used as bullet markers in text (e.g., "* Item one") should be extracted as unordered list items, NOT as paragraphs with literal asterisks. Convert markdown-style bullet asterisks to proper list objects.
- IMPORTANT: Text from the SAME logical section that spans two pages should be treated as continuous — do NOT restart paragraph numbering, list numbering, or heading levels just because a new page begins.

Return ONLY the JSON array. No explanations or markdown code blocks.