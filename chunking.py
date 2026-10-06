def chunk_text(text, chunk_size=1000, chunk_overlap=150):
    """Recursive character chunking: splits on paragraph, then line, then
    sentence, then word boundaries, so chunks stay semantically coherent
    instead of being cut mid-sentence. Adjacent chunks overlap so context
    isn't lost at the boundary.
    """
    separators = ["\n\n", "\n", ". ", " "]
    chunks = _split(text, separators, chunk_size)
    return _add_overlap(chunks, chunk_overlap)


def _split(text, separators, chunk_size):
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    separator = separators[0] if separators else ""
    remaining_separators = separators[1:]
    parts = text.split(separator) if separator else list(text)

    chunks = []
    current = ""

    for part in parts:
        candidate = current + (separator if current else "") + part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            if len(part) > chunk_size and remaining_separators:
                chunks.extend(_split(part, remaining_separators, chunk_size))
                current = ""
            else:
                current = part

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _add_overlap(chunks, chunk_overlap):
    if not chunks:
        return chunks

    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-chunk_overlap:]
        overlapped.append((prev_tail + " " + chunks[i]).strip())

    return overlapped
