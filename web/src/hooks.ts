import { useCallback, useEffect, useRef, useState, type RefObject } from "react";

export interface AudioSegmentRequest {
  key: string;
  src: string;
  start?: number;
  end?: number;
}

export function useAudioPlayer() {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playingKey, setPlayingKey] = useState<string | null>(null);
  const endRef = useRef<number | null>(null);
  const generationRef = useRef(0);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const stopPlaying = () => setPlayingKey(null);
    const onTimeUpdate = () => {
      if (endRef.current != null && audio.currentTime >= endRef.current - 0.02) {
        audio.pause();
      }
    };
    audio.addEventListener("pause", stopPlaying);
    audio.addEventListener("ended", stopPlaying);
    audio.addEventListener("timeupdate", onTimeUpdate);
    return () => {
      audio.removeEventListener("pause", stopPlaying);
      audio.removeEventListener("ended", stopPlaying);
      audio.removeEventListener("timeupdate", onTimeUpdate);
    };
  }, []);

  const pause = useCallback(() => {
    const audio = audioRef.current;
    if (audio) audio.pause();
    setPlayingKey(null);
  }, []);

  const playSegment = useCallback(
    async (request: AudioSegmentRequest) => {
      const audio = audioRef.current;
      if (!audio) return;
      if (playingKey === request.key && !audio.paused) {
        audio.pause();
        return;
      }
      const generation = ++generationRef.current;
      try {
        if (audio.dataset.src !== request.src) {
          audio.src = request.src;
          audio.dataset.src = request.src;
          if (audio.readyState < 1) {
            await new Promise<void>((resolve, reject) => {
              const timer = window.setTimeout(() => {
                cleanup();
                reject(new Error("audio metadata timed out"));
              }, 5000);
              const cleanup = () => {
                window.clearTimeout(timer);
                audio.removeEventListener("loadedmetadata", onReady);
                audio.removeEventListener("error", onError);
              };
              const onReady = () => {
                cleanup();
                resolve();
              };
              const onError = () => {
                cleanup();
                reject(new Error("audio failed to load"));
              };
              audio.addEventListener("loadedmetadata", onReady, { once: true });
              audio.addEventListener("error", onError, { once: true });
              audio.load();
            });
          }
        }
        if (generation !== generationRef.current) return;
        endRef.current = request.end ?? null;
        audio.currentTime = Math.max(0, request.start ?? 0);
        await audio.play();
        if (generation === generationRef.current) setPlayingKey(request.key);
      } catch {
        if (generation === generationRef.current) setPlayingKey(null);
      }
    },
    [playingKey],
  );

  return { audioRef, playingKey, playSegment, pause };
}

export function useElementWidth<T extends HTMLElement>(): [RefObject<T | null>, number] {
  const ref = useRef<T | null>(null);
  const [width, setWidth] = useState(640);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const update = () => {
      const next = element.getBoundingClientRect().width;
      if (next > 0) setWidth(next);
    };
    update();
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", update);
      return () => window.removeEventListener("resize", update);
    }
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return [ref, width];
}
