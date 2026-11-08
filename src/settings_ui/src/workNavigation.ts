import { ListTodo, ScrollText, StickyNote, Terminal } from "lucide-react";

// All workspaces operate on the same queue and logs.
export const workNavigation = {
  group: "작업 내역 관리",
  items: [
    { id: "work-queue", icon: ListTodo, label: "작업 큐" },
    { id: "llm-logs", icon: ScrollText, label: "LLM 로그" },
    { id: "program-logs", icon: Terminal, label: "프로그램 로그" },
    { id: "memo", icon: StickyNote, label: "메모·태그 검색" },
  ],
} as const;
export type WorkPage = typeof workNavigation.items[number]["id"];
