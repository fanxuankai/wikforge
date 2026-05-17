"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { LogOut, Settings } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAuthStore } from "@/stores/auth-store";

/**
 * 用户信息展示组件（头像、名称、设置、登出按钮）
 * 用于顶部栏
 */
export function UserMenu() {
  const router = useRouter();
  const { user, logout } = useAuthStore();

  if (!user) return null;

  const initials = getInitials(user.display_name || user.email);

  const handleLogout = () => {
    logout();
    router.push("/login");
  };

  return (
    <div className="flex items-center gap-2">
      {/* Avatar with initials */}
      <div className="flex items-center gap-2">
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-xs font-medium text-primary-foreground">
          {initials}
        </div>
        <span className="hidden text-sm font-medium sm:inline-block">
          {user.display_name || user.email}
        </span>
      </div>

      {/* Settings */}
      <Link href="/settings" passHref>
        <Button
          variant="ghost"
          size="icon"
          aria-label="账户设置"
          title="账户设置"
        >
          <Settings className="h-4 w-4" />
        </Button>
      </Link>

      {/* Logout button */}
      <Button
        variant="ghost"
        size="icon"
        onClick={handleLogout}
        aria-label="登出"
        title="登出"
      >
        <LogOut className="h-4 w-4" />
      </Button>
    </div>
  );
}

function getInitials(name: string): string {
  if (!name) return "U";

  // For email addresses, use first letter
  if (name.includes("@")) {
    return name[0].toUpperCase();
  }

  // For Chinese names, use first 1-2 characters
  if (/[\u4e00-\u9fa5]/.test(name)) {
    return name.slice(0, 2);
  }

  // For English names, use first letter of first and last name
  const parts = name.trim().split(/\s+/);
  if (parts.length >= 2) {
    return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
  }
  return name[0].toUpperCase();
}
