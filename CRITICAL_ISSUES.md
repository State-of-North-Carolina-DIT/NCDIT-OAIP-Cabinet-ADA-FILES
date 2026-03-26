# Critical Issues Analysis & Resolution

Analysis of all issues in `critical_issues.csv`, with status and remediation notes.

---

## Issues Fixed in `render_json.py` (this session)

### 1. Consecutive Heading Merger — Em-Dash Introduced Globally
**Affected files:** `near-perfect-powerpoint-*` (global), `logo-tables-shading-watermark-photos-13a3`, and many others
**CSV description:** "NEW: Linebreak replaced by em dash (Pg. 12, 15, 19, 20, 22, 26, 27... Global)"
**Root cause:** `_merge_consecutive_headings()` merged any two adjacent same-level headings unconditionally, combining independent section headings (e.g. slide title + slide sub-section) into a single heading with " — " separator.
**Fix:** `_merge_consecutive_headings()` now only merges when the **second** heading matches a date/time pattern (month names, day names, or year). This preserves legitimate date-subtitle merges (e.g. "BOARD MEETING — Wednesday, August 14, 2019") while keeping independent headings separate.

---

### 2. Ordered List Numbering Restarting After Page Breaks
**Affected files:** `scio-physical-and-environmental-protection-686b1b84`, `logo-tables-shading-watermark-photos-13a3`, `20190416-nc-911-board-minutes-approved-687248c4`, many others
**CSV description:** "When lists are separated, but the numbers continue in the latter positions of the list, the numbers start again at 1"
**Root cause (a):** The `start` attribute on `<ol>` was only calculated for numeric lists, not for letter-type (`a`, `A`) or roman numeral (`i`, `I`) lists. So `<ol type="a">` starting at "g." would still render from "a.".
**Fix (a):** Extended `start` attribute detection in `_render_list()` to cover letter and roman numeral list styles using new helper functions `_letter_to_int()`, `_roman_to_int()`, and `_list_item_ordinal()`. A list beginning with "g." now renders as `<ol type="a" start="7">`.
**Root cause (b):** `_merge_consecutive_lists()` only merged consecutive lists within the same page, not across PDF page boundaries.
**Fix (b):** Added `_merge_consecutive_lists_across_pages()` which detects when the last list on page N and the first list on page N+1 are continuations (e.g., last item "f." → first item "g.") and merges them into a single list.

---

### 3. Link Display Text Contains Appended URL
**Affected files:** `seal-image-table-6870`, and potentially others
**CSV description:** "Hyperlink 'CLICK HERE' has an href of literally 'CLICK HERE' instead of the real link from the original PDF. But then puts a 2nd link at another random place in the html, CLICK HERE, that does have the correct link."
**Root cause:** Gemini sometimes appended the URL to the link display text (e.g., "CLICK HERE https://it.nc.gov/..."). The broken-link fixer in `_deduplicate_links()` correctly set the href to the real URL, but the display text still contained the appended URL.
**Fix:** Added a pre-pass in `_deduplicate_links()` that strips any trailing URL pattern from link display text when the URL appears appended to the display text.

---

### 4. Table Caption Not Processing Markdown
**Affected files:** Any file with `**bold**` in table titles/captions
**CSV description:** "Asterisks added to HTML on heading and table heading that aren't on PDF"
**Root cause:** `_render_table()` used `escape(caption)` for the `<caption>` element, which HTML-escaped but did not apply markdown-to-HTML conversion. So `**bold**` in a caption appeared as `**bold**` in the output.
**Fix:** Changed `escape(caption)` → `_md_to_html(caption)` for both the main caption and the empty-table fallback caption.

---

### 5. Table of Contents Removal (from previous session)
**Affected files:** `gdac-legislative-report-may-2016-68610a4b`, `logo-tables-shading-watermark-photos-13a3`, `seal-imagery-11ab`, `seal-imagery-table-with-shading-colored-text-672b`, and others
**CSV description:** "Remove table of contents in favor of H2 anchor links. Table of Contents in favor of page numbers is useless."
**Fix:** Added `_remove_table_of_contents()` ADA remediation function that detects ToC headings ("Table of Contents", "Contents", "TOC", "List of Figures", etc.) and removes the heading and all following ToC entries (items ending with leader dots + page number).

---

### 6. Table `aria-label` → `<caption>` Fallback (from previous session)
**Affected files:** `seal-image-table-6870` (CSV: "Unneeded aria-label on the table")
**Fix:** `_render_table()` now falls back to `item.get("aria_label")` as a third option for caption content, after `caption` and `title`. The table's accessible name is preserved as `<caption>` since `aria-label` is stripped per the no-ARIA policy.

---

### 7. Image `role="presentation"` → `alt=""` (from previous session)
**CSV description:** "When role=presentation is removed from images, is it replaced with empty alt?"
**Fix:** `_mark_decorative_images()` now checks for `role in ("presentation", "none")` and marks those images as decorative, which renders them as `<img alt="">`.

---

### 8. `scope="col"` / `scope="row"` Restored on `<th>` (from previous session)
**CSV description:** "Why remove scope=col from `<th>`? That is useful for accessibility."
**Fix:** `_render_table()` now adds `scope="col"` to header cells in row 0 (column headers) and `scope="row"` to header cells in column 0 (row headers).

---

### 9. `<meta name="viewport">` in Rendered HTML (from previous session)
**CSV description:** "meta viewport should not be in the JSON — we want only the semantic markup, nothing that should be in `<head>`"
**Fix:** Added `<meta name="viewport" content="width=device-width, initial-scale=1">` to the HTML `<head>` template in `render_document()`. It belongs in the rendered output, not in the JSON payload.

---

## Issues NOT Fixable in `render_json.py`

These issues originate in the **Gemini extraction layer** (`extract_structured_json.py`) or are fundamental architectural constraints. They require either extraction-layer changes or manual remediation.

---

### Hallucinated/Incorrect Content
**Affected files:** `scanned-from-paper-many-pages-of-tables-6878` ("Hallucinated the incorrect price for part '185669'"), `seal-imagery-11ab` ("Hallucinated a link 'network' to go to some random site"), `gicc-2020-census-nc-factsheet-20170809-686131e8`
**Reason:** Gemini LLM hallucinations during extraction. Not detectable or correctable in the renderer.

---

### Incorrect / Missing Alt Text
**Affected files:** `gicc-tims-may-2016-686133f1`, `newsletter-with-many-images-and-formatted-text-*`, `near-perfect-powerpoint-*`, `logo-tables-shading-watermark-photos-13a3`, many others
**Reason:** Alt text is generated by Gemini during extraction. The renderer faithfully uses whatever description is in the JSON. Incorrect descriptions must be fixed in the extraction prompt or post-extraction.

---

### Paragraphs Split at PDF Page Boundaries
**Affected files:** `seal-imagery-11ab` ("All pages: Paragraphs are split because of page splits"), `newsletter-with-many-images-and-formatted-text-21fb`, `logo-tables-shading-watermark-photos-13a3`, others
**Reason:** PDF extraction processes page-by-page. Sentences that cross page boundaries appear as two separate paragraphs. Fixing this requires cross-page sentence boundary detection in the extractor.

---

### Images Out of Order / Images Duplicated (image + transcribed text)
**Affected files:** `screenshot-images-11b5`, `gicc-lgc-censussurveyresults-20200506-68839ae8`, `powerpoint-slides-*`, newsletters
**Reason:** Gemini processes images and surrounding text independently. Image placement and the decision to include both an image and its transcribed text are extraction-layer choices.

---

### Tables Split Across Pages / Table Structure Incorrect
**Affected files:** `20190416-nc-911-board-minutes-approved-687248c4` ("Nested table converted into parent table"), `seal-image-table-6870` ("Splits a table into two at the page break"), `scanned-from-paper-many-pages-of-tables-6878` ("Broke up the 2nd row")
**Reason:** Multi-page tables are not reassembled during extraction. Nested table detection is an extraction-layer capability.

---

### Links Extracted Outside Their Original Table/List Context
**Affected files:** `gicc-agenda-20160810-68613387` ("Links pulled from table and placed at bottom"), `seal-imagery-table-with-shading-colored-text-672b` ("links and email addresses duplicated at bottom"), `newsletter-with-many-images-and-formatted-text-21fb`
**Reason:** Gemini extracts hyperlinks as standalone `link` items separately from the table or paragraph they appear in. This is an extraction architecture decision. The `_deduplicate_links()` function removes links whose URL appears in surrounding paragraph text, but cannot remove links that were inside tables.

---

### Broken Video Links (YouTube home page instead of real link)
**Affected files:** `newsletter-with-many-images-and-formatted-text-117c`, `newsletter-with-many-images-and-formatted-text-21fb`
**Reason:** Gemini extracted the wrong URL for embedded video players. The extractor found the YouTube.com homepage rather than the specific video link.

---

### Missing State Crest / Logo Images in Tables
**Affected files:** `seal-imagery-table-with-shading-132c` ("Missing State Crest on all pages, showing alt text in the header rather than crest"), `scio-physical-and-environmental-protection-686b1b84`
**Reason:** The state seal image appears in a header table cell. During extraction, the image was captured as the alt text description only (no base64 image data), and `_render_image()` renders images without base64 as `<p>[Image: ...]</p>`. For images extracted inside table cells, this is a limitation of the JSON schema (images inside table cells are not directly representable).

---

### Form Fields Converted to Tables
**Affected files:** `seal-imagery-table-with-shading-132c` ("Form check blocks turned into a table"), `fillable-form-logo-imagery-bdfc`, `ifb-its-400277-2017-1102-final-686b18e9`
**Reason:** Checkbox/form content in PDFs is complex to parse. Gemini sometimes represents these as tables. This is an extraction-layer decision.

---

### Unordered List Where Ordered (Letter Bullets) Expected
**Affected files:** `scio-physical-and-environmental-protection-686b1b84` (partially fixed via `start` attribute; the cross-page merge handles the main case), others
**Remaining issue:** When Gemini explicitly extracts letter-prefixed items as `list_type: "unordered"` rather than "ordered", the renderer cannot convert them without potentially breaking bullet lists that legitimately contain "a." in item text.
**Note:** The `start` attribute fix handles lists already tagged as `ordered`; lists tagged `unordered` that appear to have letter/number prefixes would require an inference heuristic in the extraction layer.

---

### Markdown in List Items Not Rendered
**Affected files:** `gicc-smac-agenda-20210120-68839aac` (in older runs), others
**Status:** `_md_to_html()` is applied to all text content including list items. The older instances of `*italic*` appearing literally were extraction issues that appear resolved in newer JSON files.

---

### Nested Lists Missing 3rd Level
**Affected files:** `ncom-update-gicc-05-15-2014-68612fe6`, `gicc-smac-agenda-20210120-68839aac`
**Reason:** Gemini does not always extract 3rd-level list nesting. The JSON only has 2-level nesting support (items + children). Third-level nesting would need to be added to both the JSON schema and the renderer.

---

## Summary Table

| Category | Count | Status |
|----------|-------|--------|
| Fixed in render_json.py (this session) | 4 | ✅ Done |
| Fixed in render_json.py (previous session) | 5 | ✅ Done |
| Gemini extraction issues (not fixable in renderer) | 11+ | ⚠️ Extraction layer |
| Architectural limitations | 3 | ⚠️ By design |
