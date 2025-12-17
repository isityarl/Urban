import { useEffect } from "react";

export interface ToastMessage {
  id: string;
  kind: "success" | "error" | "info";
  text: string;
}

interface Props {
  toasts: ToastMessage[];
  onDismiss: (id: string) => void;
  durationMs?: number;
}

export function Toasts({ toasts, onDismiss, durationMs = 3500 }: Props) {
  useEffect(() => {
    const timers = toasts.map((t) =>
      setTimeout(() => {
        onDismiss(t.id);
      }, durationMs)
    );
    return () => {
      timers.forEach(clearTimeout);
    };
  }, [toasts, onDismiss, durationMs]);

  return (
    <div className="toast-container">
      {toasts.map((t) => (
        <div key={t.id} className={`toast toast-${t.kind}`}>
          <span>{t.text}</span>
          <button className="toast-close" aria-label="Закрыть" onClick={() => onDismiss(t.id)}>
            ×
          </button>
        </div>
      ))}
    </div>
  );
}










