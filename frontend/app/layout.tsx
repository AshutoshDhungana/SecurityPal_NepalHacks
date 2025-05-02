import type React from "react"
import type { Metadata } from "next"
import "./globals.css"
import { MainSidebar } from "@/components/main-sidebar"
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar"

export const metadata: Metadata = {
  title: "Knowledge Library Enhancement System",
  description: "Enhance your security knowledge library with AI-powered analysis",
  generator: 'v0.dev'
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="bg-background min-h-screen">
        <SidebarProvider>
          <div className="flex min-h-screen">
            {/* <MainSidebar /> */}
            <div className="flex-1 flex flex-col">
              <main className="flex-1">{children}</main>
            </div>
          </div>
        </SidebarProvider>
      </body>
    </html>
  )
}
