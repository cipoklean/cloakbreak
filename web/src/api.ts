export const API = import.meta.env.VITE_API_BASE ?? "";

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

export function apiUrl(path: string): string {
  const suffix = path.startsWith("/") ? path : `/${path}`;
  if (!API) return suffix;
  return `${trimTrailingSlash(API)}${suffix}`;
}

export function websocketUrl(path = "/ws/detect"): string {
  if (API) {
    const absolute = new URL(API, window.location.href);
    absolute.protocol = absolute.protocol === "https:" ? "wss:" : "ws:";
    absolute.pathname = `${trimTrailingSlash(absolute.pathname)}${path}`;
    absolute.search = "";
    absolute.hash = "";
    return absolute.toString();
  }
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}${path}`;
}

export function detailFromBody(body: unknown, fallback: string): string {
  if (body && typeof body === "object") {
    const detail = (body as { detail?: unknown }).detail;
    if (typeof detail === "string" && detail) return detail;
    if (Array.isArray(detail)) {
      const first = detail[0] as { msg?: unknown } | undefined;
      if (first && typeof first.msg === "string") return first.msg;
    }
    const error = (body as { error?: unknown }).error;
    if (typeof error === "string" && error) return error;
  }
  return fallback;
}

export async function readJsonResponse<T>(response: Response, fallback: string): Promise<T> {
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    // handled below through the fallback
  }
  if (!response.ok) {
    throw new Error(detailFromBody(body, fallback));
  }
  return body as T;
}

export async function getJson<T>(path: string, fallback: string): Promise<T> {
  const response = await fetch(apiUrl(path));
  return readJsonResponse<T>(response, fallback);
}

export async function postJson<T>(path: string, payload: unknown, fallback: string): Promise<T> {
  const response = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return readJsonResponse<T>(response, fallback);
}

export async function postForm<T>(path: string, form: FormData, fallback: string): Promise<T> {
  const response = await fetch(apiUrl(path), {
    method: "POST",
    body: form,
  });
  return readJsonResponse<T>(response, fallback);
}
