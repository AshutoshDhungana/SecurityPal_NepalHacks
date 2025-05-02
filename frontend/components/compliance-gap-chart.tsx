"use client"

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell, LabelList } from 'recharts'

export function ComplianceGapChart() {
  // Prepare data for Recharts
  const data = [
    {
      framework: "NIST 800-53",
      Compliant: 65,
      Gap: 35
    },
    {
      framework: "ISO 27001",
      Compliant: 82,
      Gap: 18
    },
    {
      framework: "PCI DSS",
      Compliant: 78,
      Gap: 22
    }
  ]

  return (
    <div className="w-full h-[300px]">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          layout="vertical"
          data={data}
          margin={{ top: 20, right: 30, left: 100, bottom: 20 }}
          stackOffset="expand"
          barSize={40}
          barGap={4}
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis type="number" domain={[0, 100]} tickFormatter={(value) => `${value}%`} />
          <YAxis type="category" dataKey="framework" width={90} />
          <Tooltip
            formatter={(value) => [`${value}%`, 'Coverage']}
            labelFormatter={(value) => `Framework: ${value}`}
          />
          <Legend />
          <Bar dataKey="Compliant" stackId="stack" fill="#22c55e">
            <LabelList dataKey="Compliant" position="center" fill="#ffffff" formatter={(value) => value > 15 ? `${value}%` : ''} />
          </Bar>
          <Bar dataKey="Gap" stackId="stack" fill="#ef4444">
            <LabelList dataKey="Gap" position="center" fill="#ffffff" formatter={(value) => value > 15 ? `${value}%` : ''} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
