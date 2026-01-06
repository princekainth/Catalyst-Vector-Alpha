import "./globals.css";
import type { ReactNode } from "react";
import {
  ClerkProvider,
  SignInButton,
  SignedIn,
  SignedOut,
  UserButton,
} from "@clerk/nextjs";
import Toaster from "@/components/ui/toaster";

export const metadata = {
  title: "CVA Cloud",
  description: "CVA SaaS remediation console",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <ClerkProvider>
      <html lang="en">
        <body>
          <header className="flex justify-end p-4 border-b border-white/10">
            <SignedOut>
              <SignInButton mode="modal" />
            </SignedOut>
            <SignedIn>
              <UserButton />
            </SignedIn>
          </header>
          <Toaster />
          {children}
        </body>
      </html>
    </ClerkProvider>
  );
}
