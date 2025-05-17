<template>
    <div class="app-container">
        <h1>Email Phishing Detector</h1>

        <EmailForm @analyze="handleAnalyzeEmail" />

        <div v-if="result" class="mt-6 w-full max-w-3xl p-4 rounded shadow">
            <h2 class="text-cl font-semibold mb-2">Result</h2>
            <p class="mb-2">
                <strong>Status:</strong>
                <span
                    :class="
                        result.prediction === 'PHISHING'
                            ? 'text-red-600 font-bold'
                            : 'text-green-60 font-bold'
                    "
                >
                    {{ result.prediction }}
                </span>
            </p>
            <p class="mb-4">
                <strong>Confidence:</strong>
                {{ (result.confidence * 100).toFixed(2) }}%
            </p>

            <h3 class="text-lg font-semibold mb-2">Top Influencing Words</h3>
            <ul class="space-y-1">
                <li
                    v-for="(reason, index) in result.reasons"
                    :key="index"
                    class="flex items-center justify-between border-b py-1"
                >
                    <span
                        :class="
                            reason[1] === 'increased'
                                ? 'text-red-600'
                                : 'text-green-600'
                        "
                        class="font-medium"
                    >
                        {{ reason[0] }}
                    </span>
                    <span class="text-sm italic">
                        {{
                            reason[1] === 'increased'
                                ? '↑ Increased risk'
                                : '↓ Decreased risk'
                        }}
                        ({{ reason[2].toFixed(2) }})
                    </span>
                </li>
            </ul>
        </div>
    </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import EmailForm from '../components/EmailForm.vue'
import { analyzeEmail } from '../composables/useEmailAnalysis'
import type { StructuredEmail, EmailAnalysisResult } from '../types'

const result = ref<EmailAnalysisResult | null>(null)

async function handleAnalyzeEmail(
    structured: boolean,
    payload: StructuredEmail | string
) {
    try {
        const analysis = await analyzeEmail(structured, payload)
        result.value = analysis.result.value
    } catch (err: any) {
        alert('Failed to analyze email: ' + err.message)
        result.value = null
    }
}
</script>

<style scoped>
.app-container {
    background-color: #121212;
    color: #ffffff;
    font-family: sans-serif;
    padding: 2rem;
    min-height: 100vh;
}
</style>
