"use client";

import { Bot, User } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { type ChatMessage } from "@/stores/chat-store";
import { CitationLink } from "./citation-link";
import { cn } from "@/lib/utils";

interface MessageBubbleProps {
  message: ChatMessage;
}

/**
 * 助手消息使用 Markdown 渲染 (含 GFM 表格 / 任务列表 / 删除线 / 代码块)。
 * 用户消息保持纯文本展示, 不解析 Markdown 防止误显示。
 */
function MarkdownContent({ children }: { children: string }) {
  return (
    <div
      className={cn(
        "prose prose-sm max-w-none break-words dark:prose-invert",
        // 收紧默认 prose 间距, 跟普通气泡一致
        "prose-p:my-1 prose-headings:my-2 prose-ul:my-1 prose-ol:my-1 prose-li:my-0",
        "prose-pre:my-2 prose-pre:bg-background/60 prose-pre:text-xs",
        "prose-code:before:content-none prose-code:after:content-none",
        "prose-code:rounded prose-code:bg-background/60 prose-code:px-1 prose-code:py-0.5",
        "prose-a:text-primary"
      )}
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{children}</ReactMarkdown>
    </div>
  );
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div
      className={cn(
        "flex gap-3",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-muted text-muted-foreground"
        )}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>

      {/* Content */}
      <div
        className={cn(
          "flex max-w-[80%] flex-col gap-2",
          isUser ? "items-end" : "items-start"
        )}
      >
        <div
          className={cn(
            "rounded-lg px-4 py-2.5 text-sm leading-relaxed",
            isUser
              ? "bg-primary text-primary-foreground"
              : "bg-muted text-foreground"
          )}
        >
          {isUser ? (
            <div className="whitespace-pre-wrap break-words">
              {message.content}
            </div>
          ) : (
            <MarkdownContent>{message.content}</MarkdownContent>
          )}
        </div>

        {/* Citations */}
        {message.citations && message.citations.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {message.citations.map((citation, index) => (
              <CitationLink
                key={`${citation.document_id}-${citation.chunk_index}`}
                citation={citation}
                index={index}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

interface StreamingBubbleProps {
  content: string;
}

export function StreamingBubble({ content }: StreamingBubbleProps) {
  return (
    <div className="flex gap-3">
      {/* Avatar */}
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted text-muted-foreground">
        <Bot className="h-4 w-4" />
      </div>

      {/* Content */}
      <div className="flex max-w-[80%] flex-col gap-2 items-start">
        <div className="rounded-lg bg-muted px-4 py-2.5 text-sm leading-relaxed text-foreground">
          {content ? (
            <MarkdownContent>{content}</MarkdownContent>
          ) : null}
          <span className="inline-block w-1.5 h-4 ml-0.5 bg-foreground/70 animate-pulse align-middle" />
        </div>
      </div>
    </div>
  );
}
