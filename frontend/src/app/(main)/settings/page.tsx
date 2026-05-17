"use client";

import * as React from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useToast } from "@/components/ui/toast";
import { apiClient, ApiClientError } from "@/lib/api-client";
import { useAuthStore } from "@/stores/auth-store";

export default function SettingsPage() {
  const { user, setUser } = useAuthStore();
  const { addToast } = useToast();

  const [displayName, setDisplayName] = React.useState(user?.display_name || "");
  const [savingProfile, setSavingProfile] = React.useState(false);

  const [currentPwd, setCurrentPwd] = React.useState("");
  const [newPwd, setNewPwd] = React.useState("");
  const [confirmPwd, setConfirmPwd] = React.useState("");
  const [savingPwd, setSavingPwd] = React.useState(false);

  React.useEffect(() => {
    if (user?.display_name) setDisplayName(user.display_name);
  }, [user]);

  const handleSaveProfile = async () => {
    if (!displayName.trim()) {
      addToast({ type: "error", message: "显示名不能为空" });
      return;
    }
    setSavingProfile(true);
    try {
      const updated = await apiClient.patch<{
        id: string;
        email: string;
        display_name: string | null;
      }>("/api/auth/me", { display_name: displayName.trim() });
      setUser({ ...user!, display_name: updated.display_name || "" });
      addToast({ type: "success", message: "个人信息已更新" });
    } catch (err) {
      const msg = err instanceof ApiClientError ? err.message : "保存失败";
      addToast({ type: "error", message: "保存失败", description: msg });
    } finally {
      setSavingProfile(false);
    }
  };

  const handleChangePwd = async () => {
    if (!currentPwd || !newPwd) {
      addToast({ type: "error", message: "请填写完整" });
      return;
    }
    if (newPwd.length < 8) {
      addToast({ type: "error", message: "新密码至少 8 位" });
      return;
    }
    if (newPwd !== confirmPwd) {
      addToast({ type: "error", message: "两次输入的新密码不一致" });
      return;
    }
    if (newPwd === currentPwd) {
      addToast({ type: "error", message: "新密码不能与当前密码相同" });
      return;
    }

    setSavingPwd(true);
    try {
      await apiClient.post("/api/auth/change-password", {
        current_password: currentPwd,
        new_password: newPwd,
      });
      setCurrentPwd("");
      setNewPwd("");
      setConfirmPwd("");
      addToast({ type: "success", message: "密码已修改" });
    } catch (err) {
      const msg = err instanceof ApiClientError ? err.message : "修改失败";
      addToast({ type: "error", message: "修改失败", description: msg });
    } finally {
      setSavingPwd(false);
    }
  };

  if (!user) return null;

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold">账户设置</h1>
        <p className="text-muted-foreground mt-1">
          管理你的个人信息与登录凭证
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>个人信息</CardTitle>
          <CardDescription>邮箱无法修改, 显示名可以随时调整</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label className="text-sm font-medium">邮箱</label>
            <Input value={user.email} disabled className="mt-1" />
          </div>
          <div>
            <label className="text-sm font-medium">显示名</label>
            <Input
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="你希望别人怎么称呼你"
              className="mt-1"
            />
          </div>
          <Button onClick={handleSaveProfile} disabled={savingProfile}>
            {savingProfile ? "保存中..." : "保存"}
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>修改密码</CardTitle>
          <CardDescription>
            修改后当前会话不会失效, 但建议重新登录
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label className="text-sm font-medium">当前密码</label>
            <Input
              type="password"
              value={currentPwd}
              onChange={(e) => setCurrentPwd(e.target.value)}
              className="mt-1"
              autoComplete="current-password"
            />
          </div>
          <div>
            <label className="text-sm font-medium">新密码 (至少 8 位)</label>
            <Input
              type="password"
              value={newPwd}
              onChange={(e) => setNewPwd(e.target.value)}
              className="mt-1"
              autoComplete="new-password"
            />
          </div>
          <div>
            <label className="text-sm font-medium">确认新密码</label>
            <Input
              type="password"
              value={confirmPwd}
              onChange={(e) => setConfirmPwd(e.target.value)}
              className="mt-1"
              autoComplete="new-password"
            />
          </div>
          <Button onClick={handleChangePwd} disabled={savingPwd}>
            {savingPwd ? "修改中..." : "修改密码"}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
