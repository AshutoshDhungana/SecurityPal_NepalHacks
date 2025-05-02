import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export function ComplianceRequirementsList() {
  return (
    <Tabs defaultValue="nist">
      <TabsList className="w-full">
        <TabsTrigger value="nist">NIST 800-53</TabsTrigger>
        <TabsTrigger value="iso">ISO 27001</TabsTrigger>
        <TabsTrigger value="pci">PCI DSS</TabsTrigger>
      </TabsList>

      <TabsContent value="nist" className="mt-4 space-y-4">
        <div className="rounded-md border p-4">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2">
                <div className="font-medium">AC-2 Account Management</div>
                <Badge>Access Control</Badge>
              </div>
              <div className="text-sm text-muted-foreground mt-2">
                The organization manages information system accounts, including establishing, activating, modifying,
                reviewing, disabling, and removing accounts.
              </div>
              <div className="mt-3 text-sm">
                <div className="font-medium">Key Requirements:</div>
                <ul className="list-disc pl-5 mt-1 space-y-1">
                  <li>Identify and select account types (e.g., individual, group, system)</li>
                  <li>Establish conditions for group and role membership</li>
                  <li>Specify authorized users and access privileges</li>
                  <li>Require approvals for account creation</li>
                  <li>Monitor account usage and notify account managers</li>
                  <li>Authorize access based on intended system usage</li>
                </ul>
              </div>
            </div>
            <Badge variant="outline">High Priority</Badge>
          </div>
          <div className="flex justify-end gap-2 mt-4">
            <Button variant="outline" size="sm">
              View Full Control
            </Button>
            <Button size="sm">Check Compliance</Button>
          </div>
        </div>

        <div className="rounded-md border p-4">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2">
                <div className="font-medium">AU-11 Audit Record Retention</div>
                <Badge>Audit and Accountability</Badge>
              </div>
              <div className="text-sm text-muted-foreground mt-2">
                The organization retains audit records for a specified time period to provide support for after-the-fact
                investigations of security incidents and to meet regulatory and organizational information retention
                requirements.
              </div>
              <div className="mt-3 text-sm">
                <div className="font-medium">Key Requirements:</div>
                <ul className="list-disc pl-5 mt-1 space-y-1">
                  <li>Retain audit records for defined time period</li>
                  <li>Ensure retention supports after-the-fact investigations</li>
                  <li>Align with regulatory and organizational requirements</li>
                </ul>
              </div>
            </div>
            <Badge variant="outline">Medium Priority</Badge>
          </div>
          <div className="flex justify-end gap-2 mt-4">
            <Button variant="outline" size="sm">
              View Full Control
            </Button>
            <Button size="sm">Check Compliance</Button>
          </div>
        </div>

        <div className="rounded-md border p-4">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2">
                <div className="font-medium">IR-4 Incident Handling</div>
                <Badge>Incident Response</Badge>
              </div>
              <div className="text-sm text-muted-foreground mt-2">
                The organization implements an incident handling capability for security incidents that includes
                preparation, detection and analysis, containment, eradication, and recovery.
              </div>
              <div className="mt-3 text-sm">
                <div className="font-medium">Key Requirements:</div>
                <ul className="list-disc pl-5 mt-1 space-y-1">
                  <li>Document incident handling procedures</li>
                  <li>Implement preparation, detection, analysis capabilities</li>
                  <li>Establish containment, eradication, and recovery processes</li>
                  <li>Coordinate incident handling activities with contingency planning</li>
                  <li>Incorporate lessons learned into incident response procedures</li>
                </ul>
              </div>
            </div>
            <Badge variant="outline">High Priority</Badge>
          </div>
          <div className="flex justify-end gap-2 mt-4">
            <Button variant="outline" size="sm">
              View Full Control
            </Button>
            <Button size="sm">Check Compliance</Button>
          </div>
        </div>
      </TabsContent>

      <TabsContent value="iso" className="mt-4 space-y-4">
        <div className="rounded-md border p-4">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2">
                <div className="font-medium">A.9.2 User Access Management</div>
                <Badge>Access Control</Badge>
              </div>
              <div className="text-sm text-muted-foreground mt-2">
                To ensure authorized user access and to prevent unauthorized access to systems and services.
              </div>
              <div className="mt-3 text-sm">
                <div className="font-medium">Key Requirements:</div>
                <ul className="list-disc pl-5 mt-1 space-y-1">
                  <li>Implement formal user registration and de-registration process</li>
                  <li>Implement formal user access provisioning process</li>
                  <li>Restrict and control allocation of privileged access rights</li>
                  <li>Manage secret authentication information of users</li>
                  <li>Review user access rights regularly</li>
                  <li>Remove or adjust access rights upon termination or change</li>
                </ul>
              </div>
            </div>
            <Badge variant="outline">High Priority</Badge>
          </div>
          <div className="flex justify-end gap-2 mt-4">
            <Button variant="outline" size="sm">
              View Full Control
            </Button>
            <Button size="sm">Check Compliance</Button>
          </div>
        </div>

        <div className="rounded-md border p-4">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2">
                <div className="font-medium">A.12.3 Backup</div>
                <Badge>Operations Security</Badge>
              </div>
              <div className="text-sm text-muted-foreground mt-2">To protect against loss of data.</div>
              <div className="mt-3 text-sm">
                <div className="font-medium">Key Requirements:</div>
                <ul className="list-disc pl-5 mt-1 space-y-1">
                  <li>Perform regular backups of information, software, and system images</li>
                  <li>Test backups regularly to ensure they can be restored</li>
                  <li>Protect backups against unauthorized access</li>
                  <li>Define retention periods and protection requirements</li>
                </ul>
              </div>
            </div>
            <Badge variant="outline">Medium Priority</Badge>
          </div>
          <div className="flex justify-end gap-2 mt-4">
            <Button variant="outline" size="sm">
              View Full Control
            </Button>
            <Button size="sm">Check Compliance</Button>
          </div>
        </div>
      </TabsContent>

      <TabsContent value="pci" className="mt-4 space-y-4">
        <div className="rounded-md border p-4">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2">
                <div className="font-medium">Requirement 3: Protect Stored Cardholder Data</div>
                <Badge>Data Protection</Badge>
              </div>
              <div className="text-sm text-muted-foreground mt-2">
                Protection methods such as encryption, truncation, masking, and hashing are critical components of
                cardholder data protection.
              </div>
              <div className="mt-3 text-sm">
                <div className="font-medium">Key Requirements:</div>
                <ul className="list-disc pl-5 mt-1 space-y-1">
                  <li>Keep cardholder data storage to a minimum</li>
                  <li>Do not store sensitive authentication data after authorization</li>
                  <li>Mask PAN when displayed (first six and last four digits are the maximum)</li>
                  <li>Render PAN unreadable anywhere it is stored</li>
                  <li>Protect cryptographic keys used for encryption</li>
                  <li>Document key management procedures</li>
                </ul>
              </div>
            </div>
            <Badge variant="outline">High Priority</Badge>
          </div>
          <div className="flex justify-end gap-2 mt-4">
            <Button variant="outline" size="sm">
              View Full Control
            </Button>
            <Button size="sm">Check Compliance</Button>
          </div>
        </div>

        <div className="rounded-md border p-4">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2">
                <div className="font-medium">Requirement 10: Track and Monitor Access</div>
                <Badge>Logging and Monitoring</Badge>
              </div>
              <div className="text-sm text-muted-foreground mt-2">
                Logging mechanisms and the ability to track user activities are critical in preventing, detecting, or
                minimizing the impact of a data compromise.
              </div>
              <div className="mt-3 text-sm">
                <div className="font-medium">Key Requirements:</div>
                <ul className="list-disc pl-5 mt-1 space-y-1">
                  <li>Implement audit trails to link all access to system components</li>
                  <li>Implement automated audit trails for all system components</li>
                  <li>Record specific audit trail entries for each event</li>
                  <li>Synchronize all critical system clocks</li>
                  <li>Secure audit trails so they cannot be altered</li>
                  <li>Review logs and security events daily</li>
                  <li>Retain audit trail history for at least one year</li>
                </ul>
              </div>
            </div>
            <Badge variant="outline">High Priority</Badge>
          </div>
          <div className="flex justify-end gap-2 mt-4">
            <Button variant="outline" size="sm">
              View Full Control
            </Button>
            <Button size="sm">Check Compliance</Button>
          </div>
        </div>
      </TabsContent>
    </Tabs>
  )
}
