const LEAD_KEY = 'llm_rag_chatbot_lead';
var _apiBasePromise = null;

function getApiBase() {
    if (_apiBasePromise) return _apiBasePromise;
    var origin = window.location.origin;
    var path = (window.location.pathname || '').replace(/\/$/, '') || '';
    var appBase = origin + path;
    var params = new URLSearchParams(window.location.search);
    var fromQuery = params.get('baseUrl') || params.get('api');
    if (fromQuery) {
        _apiBasePromise = Promise.resolve(fromQuery.replace(/\/$/, ''));
        return _apiBasePromise;
    }
    _apiBasePromise = fetch(appBase + '/config')
        .then(function (r) { return r.json(); })
        .then(function (d) { return (d.baseUrl || appBase).replace(/\/$/, ''); })
        .catch(function () { return appBase; });
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

var SUPPORTED_UPLOAD_EXT = ['.pdf', '.md', '.markdown'];

function fileSupported(name) {
    var ext = name.slice(name.lastIndexOf('.')).toLowerCase();
    return SUPPORTED_UPLOAD_EXT.indexOf(ext) !== -1;
}

async function uploadAttachments(apiBase) {
    var results = { uploaded: 0, failed: [] };
    for (var i = 0; i < attachments.length; i++) {
        var a = attachments[i];
        var name = a.file.name;
        if (!fileSupported(name)) {
            results.failed.push(name + ' (use PDF or Markdown for indexing)');
            continue;
        }
        var form = new FormData();
        form.append('file', a.file);
        try {
            var res = await fetch(apiBase + '/documents/upload', { method: 'POST', body: form });
            var data = res.ok ? await res.json().catch(function () { return {}; }) : {};
            if (res.ok) results.uploaded++; else results.failed.push(name + ': ' + (data.detail || res.statusText));
        } catch (err) {
            results.failed.push(name + ': ' + err.message);
        }
    }
    return results;
}

async function sendMessage() {
    var input = document.getElementById('chatInput');
    var btn = document.getElementById('sendBtn');
    var text = input.value.trim();
    var hasAttachments = attachments.length > 0;
    if (!text && !hasAttachments) return;

    if (text) addMessage(text, 'user');
    input.value = '';
    btn.disabled = true;

    var apiBase = await getApiBase();

    if (hasAttachments) {
        var uploadResults = await uploadAttachments(apiBase);
        attachments = [];
        renderAttachments();
        if (uploadResults.uploaded) {
            addMessage('Uploaded ' + uploadResults.uploaded + ' file(s) to the knowledge base.', 'bot');
        }
        if (uploadResults.failed.length) {
            addMessage('Some files could not be indexed: ' + uploadResults.failed.join('; '), 'bot');
        }
    }

    if (text) {
        try {
            var res = await fetch(apiBase + '/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: text, top_k: 8 }),
            });
            var data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Request failed');
            addMessage(data.answer, 'bot', data.sources || []);
        } catch (err) {
            addMessage('Sorry, something went wrong. ' + (err.message || ''), 'bot');
        }
    }

    btn.disabled = false;
}

var attachments = [];

function addAttachments(files) {
    if (!files || !files.length) return;
    for (var i = 0; i < files.length; i++) {
        attachments.push({ file: files[i], id: 'att-' + Date.now() + '-' + i });
    }
    renderAttachments();
}

function removeAttachment(id) {
    attachments = attachments.filter(function (a) { return a.id !== id; });
    renderAttachments();
}

function renderAttachments() {
    var strip = document.getElementById('attachmentStrip');
    if (attachments.length === 0) {
        strip.style.display = 'none';
        strip.innerHTML = '';
        return;
    }
    strip.style.display = 'flex';
    strip.innerHTML = attachments.map(function (a) {
        var name = a.file.name;
        var size = a.file.size > 1024 ? (a.file.size / 1024).toFixed(1) + ' KB' : a.file.size + ' B';
        return '<span class="attachment-chip" data-id="' + escapeHtml(a.id) + '">' +
            escapeHtml(name) + ' <small>(' + escapeHtml(size) + ')</small>' +
            ' <button type="button" class="chip-remove" aria-label="Remove">×</button></span>';
    }).join('');
    strip.querySelectorAll('.chip-remove').forEach(function (btn) {
        btn.addEventListener('click', function () {
            var chip = btn.closest('.attachment-chip');
            if (chip) removeAttachment(chip.getAttribute('data-id'));
        });
    });
}

function setupDragDrop() {
    var panel = document.getElementById('chatInputPanel');
    var wrapper = document.getElementById('chatInputWrapper');
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(function (ev) {
        panel.addEventListener(ev, function (e) {
            e.preventDefault();
            e.stopPropagation();
        });
    });
    panel.addEventListener('dragenter', function () { wrapper.classList.add('drop-over'); });
    panel.addEventListener('dragleave', function (e) {
        if (!panel.contains(e.relatedTarget)) wrapper.classList.remove('drop-over');
    });
    panel.addEventListener('drop', function (e) {
        wrapper.classList.remove('drop-over');
        if (e.dataTransfer.files.length) addAttachments(e.dataTransfer.files);
    });
}

function setupPaste() {
    document.getElementById('chatInput').addEventListener('paste', function (e) {
        if (e.clipboardData && e.clipboardData.files && e.clipboardData.files.length) {
            e.preventDefault();
            addAttachments(e.clipboardData.files);
        }
    });
}

document.getElementById('attachBtn').addEventListener('click', function () {
    document.getElementById('fileInput').click();
});
document.getElementById('fileInput').addEventListener('change', function () {
    if (this.files.length) addAttachments(this.files);
    this.value = '';
});

setupDragDrop();
setupPaste();

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
