// Regeneration jobs started in this browser session, summed per bot.
// The activity endpoint is authoritative, but it only sees a job after the
// POST lands and the next header poll runs; this keeps the 캐릭터 header count
// in step with the cards without waiting for that round trip.
type Session = { emotions: string[] };

const sessions = new Map<string, Session>();
const listeners = new Set<() => void>();

const sessionKey = (bot: string, character: string) => `${bot}\u0000${character}`;

function notify() {
  listeners.forEach(listener => listener());
}

export function setEmotionRegenerationSession(bot: string, character: string, emotions: string[]) {
  const key = sessionKey(bot, character);
  const current = sessions.get(key);
  if (current && current.emotions.join("|") === emotions.join("|")) return;
  sessions.set(key, { emotions: [...emotions] });
  notify();
}

export function clearEmotionRegenerationSession(bot: string, character: string) {
  if (sessions.delete(sessionKey(bot, character))) notify();
}

export function emotionRegenerationSessionCount(bot: string) {
  let count = 0;
  for (const [key, session] of sessions) {
    if (key.startsWith(`${bot}\u0000`)) count += session.emotions.length;
  }
  return count;
}

export function subscribeEmotionRegenerationSessions(listener: () => void) {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}
