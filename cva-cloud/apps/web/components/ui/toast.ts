export type ToastTone = "success" | "error" | "info";

export function toast(message: string, tone: ToastTone = "info") {
  if (typeof window === "undefined") {
    return;
  }
  window.dispatchEvent(new CustomEvent("cva-toast", { detail: { message, tone } }));
}
