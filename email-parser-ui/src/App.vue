<template>
    <PhishingDetector />
</template>

<script setup lang="ts">
import PhishingDetector from './page/PhishingDetector.vue'

import { inject, onMounted, onUnmounted, ref } from 'vue'

const $apm: any = inject('$apm')
const span = ref<null | { end: () => null }>(null)

onMounted(
    () => (span.value = $apm.startSpan('mount-unmount-duration', 'custom'))
)

onUnmounted(() => span.value?.end())
</script>
