"use client";

import * as React from "react";
import { ChatInterface } from "@/components/chat/chat-interface";
import { SessionList } from "@/components/chat/session-list";
import { useChatStore, type ChatMessage } from "@/stores/chat-store";
import { apiClient } from "@/lib/api-client";

export default function ChatPage() {
  const {
    sessions,
    currentSessionId,
    isLoadingSessions,
    isLoadingHistory,
    setSessions,
    setCurrentSessionId,
    setMessages,
    setIsLoadingSessions,
    setIsLoadingHistory,
    reset,
  } = useChatStore();

  // Load sessions on mount
  const loadSessions = React.useCallback(async () => {
    setIsLoadingSessions(true);
    try {
      const data = await apiClient.get<{ sessions: typeof sessions }>(
        "/api/rag/sessions"
      );
      setSessions(data.sessions || []);
    } catch {
      setSessions([]);
    } finally {
      setIsLoadingSessions(false);
    }
  }, [setSessions, setIsLoadingSessions]);

  React.useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  const handleSelectSession = async (sessionId: string) => {
    if (sessionId === currentSessionId) return;

    setCurrentSessionId(sessionId);
    setIsLoadingHistory(true);

    try {
      const data = await apiClient.get<{ messages: ChatMessage[] }>(
        `/api/rag/sessions/${sessionId}/history`
      );
      setMessages(data.messages || []);
    } catch {
      setMessages([]);
    } finally {
      setIsLoadingHistory(false);
    }
  };

  const handleNewSession = () => {
    reset();
  };

  const handleDeleteSession = async (sessionId: string) => {
    try {
      await apiClient.delete(`/api/rag/sessions/${sessionId}`);
      // 从列表中剔除已删除的会话
      setSessions(sessions.filter((s) => s.id !== sessionId));
      // 如果当前正在查看的就是它,清空 chat 区域
      if (currentSessionId === sessionId) {
        reset();
      }
    } catch (err) {
      console.error("删除会话失败", err);
    }
  };

  return (
    <div className="flex h-[calc(100vh-theme(spacing.16))] -m-6">
      {/* Session sidebar */}
      <SessionList
        sessions={sessions}
        currentSessionId={currentSessionId}
        isLoading={isLoadingSessions}
        onSelectSession={handleSelectSession}
        onNewSession={handleNewSession}
        onDeleteSession={handleDeleteSession}
      />

      {/* Chat area */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {isLoadingHistory ? (
          <div className="flex flex-1 items-center justify-center">
            <div className="text-sm text-muted-foreground">加载会话历史...</div>
          </div>
        ) : (
          <ChatInterface />
        )}
      </div>
    </div>
  );
}
