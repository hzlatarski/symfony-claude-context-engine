import { Controller } from '@hotwired/stimulus';

export default class extends Controller {
    static values = { sessionId: String, url: String };

    async startSession() {
        const r = await fetch('/api/session/start', { method: 'POST' });
        return r.json();
    }

    async endSession() {
        const r = await fetch(`/api/session/${this.sessionIdValue}/status`);
        return r.json();
    }

    async dynamic() {
        const r = await fetch(this.urlValue, { method: 'POST' });
        return r.json();
    }
}
