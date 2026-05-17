"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import {
  FileText,
  MoreHorizontal,
  FolderInput,
  Trash2,
  Tag,
  RefreshCw,
  Loader2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { apiClient } from "@/lib/api-client";

export interface Document {
  id: string;
  title: string;
  file_type: string;
  file_size: number;
  status: string;
  current_stage?: string;
  progress_percent?: number;
  tags?: string[];
  space_id?: string;
  folder_id?: string;
  created_at: string;
  updated_at: string;
}

interface DocumentListResponse {
  items: Document[];
  total: number;
  page: number;
  page_size: number;
}

const STATUS_MAP: Record<string, { label: string; className: string }> = {
  pending: { label: "待处理", className: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400" },
  parsing: { label: "解析中", className: "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400" },
  cleaning: { label: "清洗中", className: "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400" },
  chunking: { label: "分块中", className: "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400" },
  embedding: { label: "向量化中", className: "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400" },
  completed: { label: "已完成", className: "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400" },
  failed: { label: "失败", className: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400" },
};

function StatusBadge({ status }: { status: string }) {
  const config = STATUS_MAP[status] || {
    label: status,
    className: "bg-gray-100 text-gray-800",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium",
        config.className
      )}
    >
      {["parsing", "cleaning", "chunking", "embedding"].includes(status) && (
        <Loader2 className="mr-1 h-3 w-3 animate-spin" />
      )}
      {config.label}
    </span>
  );
}

interface DocumentListProps {
  spaceId?: string;
  folderId?: string;
  tagFilter?: string;
  onMoveDocument?: (doc: Document) => void;
  onDeleteDocument?: (doc: Document) => void;
  onManageTags?: (doc: Document) => void;
  refreshTrigger?: number;
}

export function DocumentList({
  spaceId,
  folderId,
  tagFilter,
  onMoveDocument,
  onDeleteDocument,
  onManageTags,
  refreshTrigger,
}: DocumentListProps) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [activeMenu, setActiveMenu] = useState<string | null>(null);
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  const pageSize = 20;

  const fetchDocuments = useCallback(async () => {
    setIsLoading(true);
    try {
      const params = new URLSearchParams();
      params.set("page", String(page));
      params.set("page_size", String(pageSize));
      if (spaceId) params.set("space_id", spaceId);
      if (folderId) params.set("folder_id", folderId);
      if (tagFilter) params.set("tag", tagFilter);

      const data = await apiClient.get<DocumentListResponse>(
        `/api/documents?${params.toString()}`
      );
      setDocuments(data.items);
      setTotal(data.total);
    } catch {
      // Silently handle errors for polling
    } finally {
      setIsLoading(false);
    }
  }, [page, spaceId, folderId, tagFilter]);

  // Initial fetch and when filters change
  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments, refreshTrigger]);

  // 点击下拉菜单以外的位置自动收起 (用 capture 阶段确保在 button onClick 之后触发)
  useEffect(() => {
    if (!activeMenu) return;
    const handler = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && target.closest("[data-row-actions]")) return;
      setActiveMenu(null);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [activeMenu]);

  // Poll every 5 seconds if any document is in processing state
  useEffect(() => {
    const hasProcessing = documents.some((d) =>
      ["pending", "parsing", "cleaning", "chunking", "embedding"].includes(
        d.status
      )
    );

    if (hasProcessing) {
      pollRef.current = setInterval(() => {
        fetchDocuments();
      }, 5000);
    }

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [documents, fetchDocuments]);

  const totalPages = Math.ceil(total / pageSize);

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString("zh-CN", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <div className="space-y-4">
      {/* Table */}
      <div className="overflow-x-auto rounded-md border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-muted/50">
              <th className="px-4 py-3 text-left font-medium">文档名称</th>
              <th className="hidden px-4 py-3 text-left font-medium sm:table-cell">
                格式
              </th>
              <th className="hidden px-4 py-3 text-left font-medium md:table-cell">
                大小
              </th>
              <th className="px-4 py-3 text-left font-medium">状态</th>
              <th className="hidden px-4 py-3 text-left font-medium lg:table-cell">
                更新时间
              </th>
              <th className="px-4 py-3 text-right font-medium">操作</th>
            </tr>
          </thead>
          <tbody>
            {isLoading && documents.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center">
                  <Loader2 className="mx-auto h-6 w-6 animate-spin text-muted-foreground" />
                  <p className="mt-2 text-sm text-muted-foreground">
                    加载中...
                  </p>
                </td>
              </tr>
            ) : documents.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center">
                  <FileText className="mx-auto h-8 w-8 text-muted-foreground" />
                  <p className="mt-2 text-sm text-muted-foreground">
                    暂无文档
                  </p>
                </td>
              </tr>
            ) : (
              documents.map((doc) => (
                <tr
                  key={doc.id}
                  className="border-b transition-colors hover:bg-muted/50"
                >
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
                      <span className="truncate font-medium">{doc.title}</span>
                    </div>
                    {doc.tags && doc.tags.length > 0 && (
                      <div className="mt-1 flex flex-wrap gap-1">
                        {doc.tags.slice(0, 3).map((tag) => (
                          <span
                            key={tag}
                            className="inline-flex rounded bg-muted px-1.5 py-0.5 text-xs"
                          >
                            {tag}
                          </span>
                        ))}
                        {doc.tags.length > 3 && (
                          <span className="text-xs text-muted-foreground">
                            +{doc.tags.length - 3}
                          </span>
                        )}
                      </div>
                    )}
                  </td>
                  <td className="hidden px-4 py-3 uppercase sm:table-cell">
                    {doc.file_type}
                  </td>
                  <td className="hidden px-4 py-3 md:table-cell">
                    {formatFileSize(doc.file_size)}
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge status={doc.status} />
                  </td>
                  <td className="hidden px-4 py-3 lg:table-cell">
                    {formatDate(doc.updated_at)}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <div className="relative inline-block" data-row-actions>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8"
                        onClick={() =>
                          setActiveMenu(
                            activeMenu === doc.id ? null : doc.id
                          )
                        }
                      >
                        <MoreHorizontal className="h-4 w-4" />
                      </Button>
                      {activeMenu === doc.id && (
                        <div className="absolute right-0 top-full z-10 mt-1 w-36 rounded-md border bg-popover p-1 shadow-md">
                          <button
                            className="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm hover:bg-accent"
                            onClick={() => {
                              onManageTags?.(doc);
                              setActiveMenu(null);
                            }}
                          >
                            <Tag className="h-3.5 w-3.5" />
                            管理标签
                          </button>
                          <button
                            className="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm hover:bg-accent"
                            onClick={() => {
                              onMoveDocument?.(doc);
                              setActiveMenu(null);
                            }}
                          >
                            <FolderInput className="h-3.5 w-3.5" />
                            移动
                          </button>
                          <button
                            className="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm text-destructive hover:bg-accent"
                            onClick={() => {
                              onDeleteDocument?.(doc);
                              setActiveMenu(null);
                            }}
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                            删除
                          </button>
                        </div>
                      )}
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">
            共 {total} 条记录
          </span>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={page <= 1}
              onClick={() => setPage((p) => p - 1)}
            >
              上一页
            </Button>
            <span className="text-sm">
              {page} / {totalPages}
            </span>
            <Button
              variant="outline"
              size="sm"
              disabled={page >= totalPages}
              onClick={() => setPage((p) => p + 1)}
            >
              下一页
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
