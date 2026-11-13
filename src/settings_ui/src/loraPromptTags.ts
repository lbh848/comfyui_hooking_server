export const promptTags = (text: string) => text.split(",").map(tag => tag.trim()).filter(Boolean);

export function matchBlockRule(tag: string, rules: string[]) {
  const value = tag.trim().toLowerCase();
  return rules.some(rule => {
    const pattern = rule.trim().toLowerCase();
    return pattern.startsWith("* ") ? value.endsWith(pattern.slice(2)) : pattern.endsWith(" *") ? value.startsWith(pattern.slice(0, -2)) : value === pattern;
  });
}

export const removeBlockedTags = (text: string, rules: string[]) => promptTags(text).filter(tag => !matchBlockRule(tag, rules)).join(", ");
