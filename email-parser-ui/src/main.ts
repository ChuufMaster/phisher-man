import { createApp, defineComponent, h } from 'vue'
import { ApmVuePlugin } from '@elastic/apm-rum-vue'
import { createRouter, createWebHashHistory } from 'vue-router'
import './style.css'
import App from './App.vue'

const Home = defineComponent({
    render: () => h('div', {}, 'home'),
})

const router = createRouter({
    history: createWebHashHistory(),
    routes: [{ path: '/', component: Home }],
})

createApp(App)
    .use(router)
    .use(ApmVuePlugin, {
        router,
        config: {
            serviceName: 'email-parser-ui',
            serverUrl: 'http://localhost:8200',
            secretToken: 'changeme',
            transactionSampleRate: 1.0,
            logLevel: 'Debug',
        },
    })
    .mount('#app')
