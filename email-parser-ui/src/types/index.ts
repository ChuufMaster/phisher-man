export interface StructuredEmail {
    sender: string
    receiver: string
    subject: string
    body: string
}

export type EmailAnalysisResult = {
    prediction: string
    confidence: number
    reasons: [string, string, number][]
}
