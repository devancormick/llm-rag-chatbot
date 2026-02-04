const LEAD_KEY = 'llm_rag_chatbot_lead';
var _apiBasePromise = null;

function getApiBase() {
    if (_apiBasePromise) return _apiBasePromise;
    var origin = window.location.origin;
    var params = new URLSearchParams(window.location.search);
    var fromQuery = params.get('baseUrl') || params.get('api');
    if (fromQuery) {
        _apiBasePromise = Promise.resolve(fromQuery.replace(/\/$/, ''));
        return _apiBasePromise;
    }
    _apiBasePromise = fetch(origin + '/config')
        .then(function (r) { return r.json(); })
        .then(function (d) { return (d.baseUrl || origin).replace(/\/$/, ''); })
        .catch(function () { return origin; });
    return _apiBasePromise;
}

function getStoredLead() {
    try {
        return JSON.parse(localStorage.getItem(LEAD_KEY));
    } catch {
        return null;
    }
}

function setStoredLead(lead) {
    localStorage.setItem(LEAD_KEY, JSON.stringify(lead));
}

function showLeadModal() {
    document.getElementById('leadModal').classList.add('active');
}

function hideLeadModal() {
    document.getElementById('leadModal').classList.remove('active');
}

async function registerLead(apiBase, email, name, company) {
    const res = await fetch(apiBase + '/leads', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, name: name || null, company: company || null }),
    });
    if (!res.ok) throw new Error('Failed to register');
    return res.json();
}

document.getElementById('leadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    var apiBase = await getApiBase();
    const email = document.getElementById('leadEmail').value.trim();
    const name = document.getElementById('leadName').value.trim();
    const company = document.getElementById('leadCompany').value.trim();

    try {
        const data = await registerLead(apiBase, email, name, company);
        setStoredLead({ email, name, company, id: data.lead_id });
        hideLeadModal();
    } catch (err) {
        alert('Could not register. Please try again.');
    }
});

function escapeHtml(s) {
    if (s == null) return '';
    var div = document.createElement('div');
    div.textContent = s;
    return div.innerHTML;
}

function addMessage(text, role, sources = []) {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = 'message ' + escapeHtml(role);

    var safe = escapeHtml(text).replace(/\n/g, '<br>');
    if (sources && sources.length > 0) {
        safe += '<div class="sources">Sources: ' + sources.map(function (s) { return escapeHtml(s.source || 'N/A'); }).join(', ') + '</div>';
    }
    div.innerHTML = safe;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

async function sendMessage() {
    const input = document.getElementById('chatInput');
    const btn = document.getElementById('sendBtn');
    const text = input.value.trim();
    if (!text) return;

    addMessage(text, 'user');
    input.value = '';
    btn.disabled = true;

    var apiBase = await getApiBase();
    try {
        const res = await fetch(apiBase + '/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: text, top_k: 5 }),
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Request failed');

        addMessage(data.answer, 'bot', data.sources || []);
    } catch (err) {
        addMessage('Sorry, something went wrong. ' + (err.message || ''), 'bot');
    } finally {
        btn.disabled = false;
    }
}

document.getElementById('sendBtn').addEventListener('click', sendMessage);
document.getElementById('chatInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

getApiBase().then(function () {
    if (!getStoredLead()) {
        showLeadModal();
    } else {
        addMessage('Hi! How can I help you today?', 'bot');
    }
});
