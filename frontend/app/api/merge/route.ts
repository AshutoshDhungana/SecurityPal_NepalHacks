import { NextResponse } from "next/server"

export async function POST(request: Request) {
  try {
    const { entries, keepFields } = await request.json()

    if (!entries || entries.length < 2 || !keepFields) {
      return NextResponse.json(
        { error: "Invalid request. Must provide at least 2 entries and fields to keep." },
        { status: 400 },
      )
    }

    // In a real implementation, this would update the database
    // For demo purposes, we'll just return a success message

    // Create the merged entry
    const mergedEntry = {
      product_id: entries[0].product_id,
      cqid: entries[0].cqid, // Keep the ID of the first entry
      created_at: new Date().toISOString(), // Set to current date
      category: entries[0].category,
      deleted_at: null,
      question: keepFields.question === 0 ? entries[0].question : entries[1].question,
      answer: keepFields.answer === 0 ? entries[0].answer : entries[1].answer,
      details: keepFields.details === 0 ? entries[0].details : entries[1].details,
    }

    return NextResponse.json({
      success: true,
      message: "Entries merged successfully",
      mergedEntry,
    })
  } catch (error) {
    console.error("Error merging entries:", error)
    return NextResponse.json({ error: "Failed to merge entries" }, { status: 500 })
  }
}
