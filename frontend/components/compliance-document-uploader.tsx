"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { FileUploader } from "@/components/file-uploader"

export function ComplianceDocumentUploader() {
  const [frameworkType, setFrameworkType] = useState("")
  const [frameworkVersion, setFrameworkVersion] = useState("")
  const [customName, setCustomName] = useState("")

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-2">
          <Label htmlFor="framework-type">Framework Type</Label>
          <Select value={frameworkType} onValueChange={setFrameworkType}>
            <SelectTrigger id="framework-type">
              <SelectValue placeholder="Select framework" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="nist">NIST</SelectItem>
              <SelectItem value="iso27001">ISO 27001</SelectItem>
              <SelectItem value="pci-dss">PCI DSS</SelectItem>
              <SelectItem value="hipaa">HIPAA</SelectItem>
              <SelectItem value="gdpr">GDPR</SelectItem>
              <SelectItem value="soc2">SOC 2</SelectItem>
              <SelectItem value="custom">Custom Framework</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {frameworkType === "custom" ? (
          <div className="space-y-2">
            <Label htmlFor="custom-name">Custom Framework Name</Label>
            <Input
              id="custom-name"
              value={customName}
              onChange={(e) => setCustomName(e.target.value)}
              placeholder="Enter framework name"
            />
          </div>
        ) : (
          <div className="space-y-2">
            <Label htmlFor="framework-version">Framework Version</Label>
            <Input
              id="framework-version"
              value={frameworkVersion}
              onChange={(e) => setFrameworkVersion(e.target.value)}
              placeholder="e.g., 800-53 Rev 5, 2013"
            />
          </div>
        )}
      </div>

      <Card className="p-4 bg-muted/50">
        <div className="text-sm font-medium mb-2">Framework Details</div>
        <div className="text-sm text-muted-foreground mb-4">
          {frameworkType === "nist" && (
            <>
              <p>
                NIST Special Publication 800-53 provides security and privacy controls for federal information systems
                and organizations.
              </p>
              <p className="mt-1">Upload the full document or specific control families relevant to your analysis.</p>
            </>
          )}
          {frameworkType === "iso27001" && (
            <>
              <p>ISO 27001 is an international standard for information security management systems (ISMS).</p>
              <p className="mt-1">Upload the full standard document or specific control annexes.</p>
            </>
          )}
          {frameworkType === "pci-dss" && (
            <>
              <p>
                PCI DSS (Payment Card Industry Data Security Standard) is a set of security standards for organizations
                that handle credit card information.
              </p>
              <p className="mt-1">Upload the full standard document or specific requirement sections.</p>
            </>
          )}
          {frameworkType === "hipaa" && (
            <>
              <p>
                HIPAA (Health Insurance Portability and Accountability Act) sets standards for protecting sensitive
                patient health information.
              </p>
              <p className="mt-1">Upload the Security Rule, Privacy Rule, or other relevant documentation.</p>
            </>
          )}
          {frameworkType === "gdpr" && (
            <>
              <p>GDPR (General Data Protection Regulation) is a regulation on data protection and privacy in the EU.</p>
              <p className="mt-1">Upload the full regulation or specific articles relevant to your analysis.</p>
            </>
          )}
          {frameworkType === "soc2" && (
            <>
              <p>SOC 2 defines criteria for managing customer data based on five "trust service principles".</p>
              <p className="mt-1">Upload the trust service criteria or specific control documentation.</p>
            </>
          )}
          {frameworkType === "custom" && (
            <>
              <p>Upload your custom compliance framework documentation.</p>
              <p className="mt-1">
                Ensure the document clearly outlines requirements and controls for proper analysis.
              </p>
            </>
          )}
          {!frameworkType && <p>Select a framework type to see more information and upload relevant documents.</p>}
        </div>
      </Card>

      <FileUploader
        acceptedFileTypes=".pdf,.docx,.txt"
        maxFileSizeMB={20}
        endpoint={`/api/import/compliance/${frameworkType}`}
        multiple={true}
      />
    </div>
  )
}
