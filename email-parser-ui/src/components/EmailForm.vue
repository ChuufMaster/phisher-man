<template>
    <div>
        <label>
            <div>
                <input type="checkbox" v-model="structuredInput" />
                Use structured input
            </div>
        </label>

        <div v-if="structuredInput">
            <input v-model="email.sender" placeholder="Sender" />
            <input v-model="email.receiver" placeholder="Receiver" />
            <input v-model="email.subject" placeholder="Subject" />
            <textarea
                v-model="email.body"
                placeholder="Body"
                class="w-full max-w-3xl h-60 p-4 border rounded-lg resize-nonoe shadow-sm"
            ></textarea>
        </div>
        <div v-else>
            <textarea
                v-model="rawEmail"
                placeholder="Paste raw email here..."
            ></textarea>
        </div>

        <button @click="handleAnalyze" :disabled="!isValid">Analyze</button>
    </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import type { StructuredEmail } from '../types'

const emit = defineEmits<{
    (e: 'analyze', structured: boolean, payload: StructuredEmail | string): void
}>()
const structuredInput = ref(false)
const rawEmail = ref('')
const email = ref<StructuredEmail>({
    sender: '',
    receiver: '',
    subject: '',
    body: '',
})

function handleAnalyze() {
    if (!isValid) return alert('Invalid input detected')
    const payload = structuredInput.value ? email.value : rawEmail.value
    emit('analyze', structuredInput.value, payload)
}

const isValid = computed(() => {
    if (structuredInput.value) {
        return (
            email.value.sender.trim().length > 3 &&
            email.value.receiver.trim().length > 3 &&
            email.value.subject.trim().length > 1 &&
            email.value.body.trim().length > 10
        )
    } else {
        return rawEmail.value.trim().length > 20
    }
})
</script>
