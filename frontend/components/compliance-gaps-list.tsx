import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

export function ComplianceGapsList() {
  return (
    <div className="space-y-6">
      <div className="rounded-md border p-4">
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2">
              <div className="font-medium">Data Retention Policy Gap</div>
              <Badge variant="destructive">High Priority</Badge>
            </div>
            <div className="text-sm text-muted-foreground mt-2">
              Current data retention policies do not fully address the requirements specified in NIST 800-53 AU-11 and
              GDPR Article 17.
            </div>

            <div className="mt-4 grid gap-4 md:grid-cols-2">
              <div className="rounded-md bg-muted/50 p-3">
                <div className="text-sm font-medium">Compliance Requirement</div>
                <div className="text-sm mt-1">
                  NIST 800-53 AU-11 requires organizations to retain audit records for a defined period to support
                  after-the-fact investigations and meet regulatory requirements.
                </div>
              </div>

              <div className="rounded-md bg-red-50 p-3">
                <div className="text-sm font-medium text-red-800">Current Status</div>
                <div className="text-sm mt-1 text-red-700">
                  Knowledge library entry "Data Retention Policy" mentions retention but does not specify timeframes or
                  categories of data.
                </div>
              </div>
            </div>

            <div className="mt-4 text-sm">
              <div className="font-medium">Gap Details:</div>
              <ul className="list-disc pl-5 mt-1 space-y-1">
                <li>No specific retention periods defined for different data categories</li>
                <li>No process documented for data deletion after retention period</li>
                <li>No exceptions process for legal holds or regulatory requirements</li>
                <li>No automated enforcement of retention policies</li>
              </ul>
            </div>
          </div>
        </div>
        <div className="flex justify-end gap-2 mt-4">
          <Button variant="outline" size="sm">
            Ignore
          </Button>
          <Button size="sm">Address Gap</Button>
        </div>
      </div>

      <div className="rounded-md border p-4">
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2">
              <div className="font-medium">Access Control Documentation Gap</div>
              <Badge variant="destructive">High Priority</Badge>
            </div>
            <div className="text-sm text-muted-foreground mt-2">
              Current access control documentation does not meet the requirements specified in ISO 27001 A.9.2 and NIST
              800-53 AC-2.
            </div>

            <div className="mt-4 grid gap-4 md:grid-cols-2">
              <div className="rounded-md bg-muted/50 p-3">
                <div className="text-sm font-medium">Compliance Requirement</div>
                <div className="text-sm mt-1">
                  ISO 27001 A.9.2 requires formal user access management processes including registration, provisioning,
                  privileged access management, and regular reviews.
                </div>
              </div>

              <div className="rounded-md bg-red-50 p-3">
                <div className="text-sm font-medium text-red-800">Current Status</div>
                <div className="text-sm mt-1 text-red-700">
                  Knowledge library mentions access controls but lacks formal documentation of processes, especially for
                  privileged access management and regular reviews.
                </div>
              </div>
            </div>

            <div className="mt-4 text-sm">
              <div className="font-medium">Gap Details:</div>
              <ul className="list-disc pl-5 mt-1 space-y-1">
                <li>No formal user registration and de-registration process documented</li>
                <li>Privileged access management procedures not defined</li>
                <li>No evidence of regular access rights reviews</li>
                <li>No documented process for removing access upon termination</li>
              </ul>
            </div>
          </div>
        </div>
        <div className="flex justify-end gap-2 mt-4">
          <Button variant="outline" size="sm">
            Ignore
          </Button>
          <Button size="sm">Address Gap</Button>
        </div>
      </div>

      <div className="rounded-md border p-4">
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2">
              <div className="font-medium">Incident Response Plan Gap</div>
              <Badge variant="destructive">Medium Priority</Badge>
            </div>
            <div className="text-sm text-muted-foreground mt-2">
              Current incident response documentation does not meet the requirements specified in NIST 800-53 IR-4 and
              ISO 27001 A.16.
            </div>

            <div className="mt-4 grid gap-4 md:grid-cols-2">
              <div className="rounded-md bg-muted/50 p-3">
                <div className="text-sm font-medium">Compliance Requirement</div>
                <div className="text-sm mt-1">
                  NIST 800-53 IR-4 requires organizations to implement an incident handling capability including
                  preparation, detection, analysis, containment, eradication, and recovery.
                </div>
              </div>

              <div className="rounded-md bg-red-50 p-3">
                <div className="text-sm font-medium text-red-800">Current Status</div>
                <div className="text-sm mt-1 text-red-700">
                  Knowledge library mentions incident response but lacks detailed procedures for each phase of incident
                  handling.
                </div>
              </div>
            </div>

            <div className="mt-4 text-sm">
              <div className="font-medium">Gap Details:</div>
              <ul className="list-disc pl-5 mt-1 space-y-1">
                <li>No documented incident classification system</li>
                <li>Containment strategies not defined for different incident types</li>
                <li>No formal process for incorporating lessons learned</li>
                <li>Incident response roles and responsibilities not clearly defined</li>
              </ul>
            </div>
          </div>
        </div>
        <div className="flex justify-end gap-2 mt-4">
          <Button variant="outline" size="sm">
            Ignore
          </Button>
          <Button size="sm">Address Gap</Button>
        </div>
      </div>

      <div className="rounded-md border p-4">
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2">
              <div className="font-medium">Backup and Recovery Gap</div>
              <Badge variant="destructive">Medium Priority</Badge>
            </div>
            <div className="text-sm text-muted-foreground mt-2">
              Current backup procedures do not fully meet the requirements specified in ISO 27001 A.12.3.
            </div>

            <div className="mt-4 grid gap-4 md:grid-cols-2">
              <div className="rounded-md bg-muted/50 p-3">
                <div className="text-sm font-medium">Compliance Requirement</div>
                <div className="text-sm mt-1">
                  ISO 27001 A.12.3 requires regular backups of information, software, and system images, with testing to
                  ensure they can be restored when needed.
                </div>
              </div>

              <div className="rounded-md bg-red-50 p-3">
                <div className="text-sm font-medium text-red-800">Current Status</div>
                <div className="text-sm mt-1 text-red-700">
                  Knowledge library mentions backups but lacks evidence of regular testing and verification of backup
                  integrity.
                </div>
              </div>
            </div>

            <div className="mt-4 text-sm">
              <div className="font-medium">Gap Details:</div>
              <ul className="list-disc pl-5 mt-1 space-y-1">
                <li>No documented schedule for backup testing</li>
                <li>No evidence of regular restoration tests</li>
                <li>Backup protection measures not fully documented</li>
                <li>Retention periods for backups not clearly defined</li>
              </ul>
            </div>
          </div>
        </div>
        <div className="flex justify-end gap-2 mt-4">
          <Button variant="outline" size="sm">
            Ignore
          </Button>
          <Button size="sm">Address Gap</Button>
        </div>
      </div>
    </div>
  )
}
