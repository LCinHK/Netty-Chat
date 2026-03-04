import type { Metadata } from "next";
import "./globals.css";
import { ReactNode, Suspense } from "react";

export const metadata: Metadata = {
  title: "ECEasy",
  description:
    "ECEasy: Your computer network co-pilot. An interactive, intelligent assistant tailored for computer network education.",
  icons: {
    icon: "/2.svg",
  },
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="/2.svg" type="image/svg+xml" />
        <title>Your App Title</title>
      </head>
      <body>
        <Suspense fallback={null}>{children}</Suspense>
      </body>
    </html>
  );
}
