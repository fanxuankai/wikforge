"use client";

import * as React from "react";
import { MessageSquarePlus, MessageSquare, Loader2, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { type ChatSession } from "@/stores/chat-store";
import { cn } from "@/lib/utils";

interface SessionListProps {
  sessions: ChatSession[];
  currentSessionId: string | null;
  isLoading: boolean;
  onSelectSession: (sessionId: string) => void;
  onNewSession: () => void;
  onDeleteSession?: (sessionId: string) => void | Promise<void>;
}

export function SessionList({
  sessions,
  currentSessionId,
  isLoading,
  onSelectSession,
  onNewSession,
  onDeleteSession,
}: SessionListProps) {
  const formatTime = (raw: string | number | undefined | null) => {
    if (raw === undefined || raw === null || raw === "") return "";

    // 兼容 ISO 字符串 / Unix 秒 / Unix 毫秒
    let date: Date;
    if (typeof raw === "number") {
      date = new Date(raw < 1e12 ? raw * 1000 : raw);
    } else {
      const asNum = Number(raw);
      if (!Number.isNaN(asNum) && raw.length <= 13) {
        date = new Date(asNum < 1e12 ? asNum * 1000 : asNum);
      } else {
        date = new Date(raw);
      }
    }

    if (isNaN(date.getTime())) return "";

    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return "刚刚";
    if (diffMins < 60) return `${diffMins} 分钟前`;
    if (diffHours < 24) return `${diffHours} 小时前`;
    if (diffDays < 7) return `${diffDays} 天前`;
    return date.toLocaleDateString("zh-CN");
  };

  const handleDelete = (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation();
    if (!onDeleteSession) return;
    if (!confirm("确认删除这个会话?")) return;
    onDeleteSession(sessionId);
  };

  return (
    <div className="flex h-full w-64 flex-col border-r">
      {/* Header */}
      <div className="flex items-center justify-between border-b px-4 py-3">
        <h3 className="text-sm font-medium">会话列表</h3>
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8"
          onClick={onNewSession}
          title="新建会话"
        >
          <MessageSquarePlus className="h-4 w-4" />
        </Button>
      </div>

      {/* Session list */}
      <ScrollArea className="flex-1">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
          </div>
        ) : sessions.length === 0 ? (
          <div className="px-4 py-8 text-center text-sm text-muted-foreground">
            暂无会话记录
          </div>
        ) : (
          <div className="space-y-1 p-2">
            {sessions.map((session) => {
              const timeText = formatTime(session.last_active_at);
              const isActive = currentSessionId === session.id;
              return (
                <div
                  key={session.id}
                  className={cn(
                    "group relative flex cursor-pointer items-start gap-2 rounded-md px-3 py-2 text-left text-sm transition-colors hover:bg-accent",
                    isActive && "bg-accent"
                  )}
                  onClick={() => onSelectSession(session.id)}
                  role="button"
                  tabIndex={0}
                >
                  <MessageSquare className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm">
                      {session.preview || "新会话"}
                    </div>
                    {timeText && (
                      <div className="text-xs text-muted-foreground">
                        {timeText}
                      </div>
                    )}
                  </div>
                  {onDeleteSession && (
                    <button
                      onClick={(e) => handleDelete(e, session.id)}
                      className="ml-1 rounded p-1 text-muted-foreground opacity-0 transition-opacity hover:bg-destructive/10 hover:text-destructive group-hover:opacity-100"
                      title="删除会话"
                      aria-label="删除会话"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </ScrollArea>
    </div>
  );
}
