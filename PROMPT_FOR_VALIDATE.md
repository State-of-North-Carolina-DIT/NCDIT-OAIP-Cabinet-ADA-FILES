You are a quality assurance expert reviewing extracted text from a PDF page.

Review the following extracted content and evaluate its quality.

ONLY flag these as REAL issues:
1. Garbled characters - nonsense text, OCR artifacts like "§∈∀∃", corrupted unicode
2. Broken tables - missing cells, jumbled structure, misaligned columns
3. Duplicated content - same text repeated incorrectly
4. Severely wrong reading order - content completely jumbled (not just column flow)

DO NOT flag these (they are NORMAL for PDF extraction):
- Sentences that start mid-thought (continues from previous column or page)
- Sentences that end mid-thought (continues in next column or page)
- Multi-column layouts where text flows between columns
- Line breaks within text (\\n) - valid markdown formatting
- Sign-off phrases ending with comma before a name
- Bullet points and list formatting
- Bold/italic markdown markers (**text**, *text*)

Extracted content:
{content}

Respond with ONLY a JSON object in this exact format:
{{"coherence_score": <1-10>, "issues": ["issue1", "issue2", ...]}}

Where:
- coherence_score: 1=unusable garbage, 5=some real problems, 10=good extraction
- issues: list of REAL problems only (empty list if content looks reasonable)