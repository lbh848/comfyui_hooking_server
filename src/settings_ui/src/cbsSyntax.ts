export type CbsPart = { text: string; syntax: boolean };

export function analyzeCbsSyntax(text: string): { parts: CbsPart[]; content: string } {
  const parts: CbsPart[] = [];
  let cursor = 0;
  while (cursor < text.length) {
    const start = text.indexOf("{{", cursor);
    if (start < 0) break;
    let depth = 1;
    let end = start + 2;
    // Conditions can contain multiple nested expressions. A flat regex stops
    // at the first inner closing pair and leaves part of the condition behind.
    while (end < text.length && depth > 0) {
      if (text.startsWith("{{", end)) { depth++; end += 2; }
      else if (text.startsWith("}}", end)) { depth--; end += 2; }
      else end++;
    }
    // Preserve an unfinished expression rather than treating its trailing prose
    // as syntax. Only complete expressions are marked for removal.
    if (depth > 0) break;
    if (start > cursor) parts.push({ text: text.slice(cursor, start), syntax: false });
    parts.push({ text: text.slice(start, end), syntax: true });
    cursor = end;
  }
  if (cursor < text.length) parts.push({ text: text.slice(cursor), syntax: false });
  return { parts, content: parts.filter(part => !part.syntax).map(part => part.text).join("") };
}
