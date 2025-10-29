"use client";
import * as React from "react";
import { useEffect, useRef, useState, type ReactNode, type RefObject } from "react";
import { ArrowUpRightIcon } from "lucide-react";
import { cn } from "@/lib/utils";

// Lightweight local mouse-tracking hook that returns element-relative coordinates
function useMouse(): [
  { elementX: number | null; elementY: number | null },
  RefObject<HTMLDivElement>
] {
  const ref = useRef<HTMLDivElement>(null);
  const [coords, setCoords] = useState<{ elementX: number | null; elementY: number | null }>({
    elementX: null,
    elementY: null,
  });

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const handleMove = (e: MouseEvent) => {
      const rect = el.getBoundingClientRect();
      setCoords({
        elementX: e.clientX - rect.left,
        elementY: e.clientY - rect.top,
      });
    };
    const handleLeave = () => setCoords({ elementX: null, elementY: null });

    el.addEventListener("mousemove", handleMove);
    el.addEventListener("mouseleave", handleLeave);
    return () => {
      el.removeEventListener("mousemove", handleMove);
      el.removeEventListener("mouseleave", handleLeave);
    };
  }, []);

  return [coords, ref];
}

export const MainMenusGradientCard = ({
  title,
  description,
  withArrow = false,
  circleSize = 400,
  className,
  children,
  size = "md",
}: {
  title: string;
  description?: string;
  withArrow?: boolean;
  circleSize?: number;
  children?: ReactNode;
  className?: string;
  size?: "sm" | "md" | "lg";
}) => {
  const [mouse, parentRef] = useMouse();

  return (
    <div
      className="group relative transform-gpu overflow-hidden rounded-[20px] bg-white/10 p-2 transition-transform hover:scale-[1.01] active:scale-90"
      ref={parentRef}
    >
      {withArrow && (
        <ArrowUpRightIcon className="absolute top-2 right-2 z-10 size-5 translate-y-4 text-neutral-700 opacity-0 transition-all group-hover:translate-y-0 group-hover:opacity-100 dark:text-neutral-300 " />
      )}
      <div
        className={cn(
          "-translate-x-1/2 -translate-y-1/2 absolute transform-gpu rounded-full transition-transform duration-500 group-hover:scale-[3]",
          mouse.elementX === null || mouse.elementY === null
            ? "opacity-0"
            : "opacity-100",
        )}
        style={{
          maskImage: `radial-gradient(${circleSize / 2}px circle at center, white, transparent)`,
          width: `${circleSize}px`,
          height: `${circleSize}px`,
          left: `${mouse.elementX}px`,
          top: `${mouse.elementY}px`,
          background: "#4520d7",
          filter: "drop-shadow(0 0 6px #2400b8)",
        }}
      />
      <div className="absolute inset-px rounded-[19px] bg-neutral-100/80 dark:bg-neutral-900/80" />
      {children && (
        <div
          className={cn(
            "grid relative h-40 place-content-center overflow-hidden rounded-[15px] border-white bg-white/70 dark:border-neutral-950 dark:bg-black/50",
            className,
          )}
        >
          {children}
        </div>
      )}
      <div className="relative px-4 pt-4 pb-2">
        <h3 className="font-semibold text-lg text-neutral-800 dark:text-neutral-300">
          {title}
        </h3>
        {description && (
          <p className="mt-2 text-neutral-600 dark:text-neutral-400">
            {description}
          </p>
        )}
      </div>
    </div>
  );
};
