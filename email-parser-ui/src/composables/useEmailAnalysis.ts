import { ref } from 'vue'
import { sanitize } from './useSanitizer'
import type { EmailAnalysisResult, StructuredEmail } from '../types'

// export function useEmailAnalysis() {
const result = ref<EmailAnalysisResult | null>(null)

export const analyzeEmail = async (
    structuredInput: boolean,
    payload: StructuredEmail | string
) => {
    try {
        const endpoint = structuredInput ? '/api/predict' : '/api/predict_raw'

        const headers = {
            'Content-Type': structuredInput ? 'application/json' : 'text/palin',
        }

        const body = structuredInput
            ? JSON.stringify(payload)
            : (payload as string)

        const response = await fetch(endpoint, {
            method: 'POST',
            headers,
            body,
        })

        if (!response.ok)
            throw new Error(`Server responded with ${response.status}`)

        const data = await response.json()
        result.value = {
            prediction: sanitize(data.prediction),
            confidence: data.confidence,
            reasons: data.reasons.map((r: [string, string, number]) => [
                sanitize(r[0]),
                sanitize(r[1]),
                r[2],
            ]),
        }
    } catch (err: any) {
        console.error('Error analyzing email: ', err)
        alert('Error analyzing email: ' + err)
    }
    console.log('here')
    return { result, analyzeEmail }
}
