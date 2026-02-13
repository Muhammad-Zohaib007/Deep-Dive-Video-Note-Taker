/* Deep-Dive Video Note Taker - Frontend JavaScript */

const API_BASE = '/api';
let currentVideoId = null;
let pollInterval = null;

// ============================================================
// Video Processing
// ============================================================

async function processVideo() {
    const url = document.getElementById('video-url').value.trim();
    if (!url) {
        alert('Please enter a video URL.');
        return;
    }

    const whisperModel = document.getElementById('whisper-model').value;
    const ollamaModel = document.getElementById('ollama-model').value;

    const btn = document.getElementById('process-btn');
    btn.disabled = true;
    btn.textContent = 'Processing...';

    showProgress();

    try {
        const res = await fetch(`${API_BASE}/process`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                url: url,
                whisper_model: whisperModel,
                ollama_model: ollamaModel,
            }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Processing failed');
        }

        const job = await res.json();
        pollJobStatus(job.job_id);
    } catch (e) {
        alert('Error: ' + e.message);
        hideProgress();
        btn.disabled = false;
        btn.textContent = 'Process';
    }
}

async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const whisperModel = document.getElementById('whisper-model').value;
    const ollamaModel = document.getElementById('ollama-model').value;

    const btn = document.getElementById('process-btn');
    btn.disabled = true;
    btn.textContent = 'Uploading...';

    showProgress();

    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('whisper_model', whisperModel);
        formData.append('ollama_model', ollamaModel);

        const res = await fetch(`${API_BASE}/process/upload`, {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Upload failed');
        }

        const job = await res.json();
        pollJobStatus(job.job_id);
    } catch (e) {
        alert('Error: ' + e.message);
        hideProgress();
        btn.disabled = false;
        btn.textContent = 'Process';
    }
}

// ============================================================
// Job Polling
// ============================================================

function pollJobStatus(jobId) {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
        try {
            const res = await fetch(`${API_BASE}/status/${jobId}`);
            if (!res.ok) {
                console.error('Poll request failed:', res.status, res.statusText);
                return; // Retry on next interval
            }
            const job = await res.json();

            updateProgressUI(job);

            if (job.status === 'completed') {
                clearInterval(pollInterval);
                pollInterval = null;
                currentVideoId = job.video_id;
                await loadNotes(job.video_id);
                await refreshLibrary();
                hideProgress();
                document.getElementById('process-btn').disabled = false;
                document.getElementById('process-btn').textContent = 'Process';
            } else if (job.status === 'failed') {
                clearInterval(pollInterval);
                pollInterval = null;
                alert('Processing failed: ' + (job.error || 'Unknown error'));
                hideProgress();
                document.getElementById('process-btn').disabled = false;
                document.getElementById('process-btn').textContent = 'Process';
            }
        } catch (e) {
            console.error('Poll error:', e);
        }
    }, 2000);
}

// ============================================================
// Progress UI
// ============================================================

const STAGE_NAMES = {
    audio_extraction: 'Audio Extraction',
    transcription: 'Transcription',
    embedding: 'Embedding',
    generation: 'Note Generation',
};

function showProgress() {
    const section = document.getElementById('progress-section');
    section.style.display = 'block';
    document.getElementById('results-section').style.display = 'none';
}

function hideProgress() {
    document.getElementById('progress-section').style.display = 'none';
}

function updateProgressUI(job) {
    const container = document.getElementById('progress-stages');
    container.innerHTML = '';

    for (const stage of job.stages || []) {
        const div = document.createElement('div');
        div.className = `progress-stage ${stage.status}`;

        let icon = '&#9711;'; // pending circle
        if (stage.status === 'processing') icon = '&#9881;'; // gear
        else if (stage.status === 'completed') icon = '&#10003;'; // check
        else if (stage.status === 'failed') icon = '&#10007;'; // X

        div.innerHTML = `
            <span class="icon">${icon}</span>
            <span>${STAGE_NAMES[stage.stage] || stage.stage}</span>
            <span style="margin-left:auto;font-size:0.75rem;color:var(--text-dim)">${stage.detail || ''}</span>
        `;
        container.appendChild(div);
    }
}

// ============================================================
// Load & Display Notes
// ============================================================

async function loadNotes(videoId) {
    currentVideoId = videoId;

    try {
        const [notesRes, transcriptRes] = await Promise.all([
            fetch(`${API_BASE}/notes/${videoId}`),
            fetch(`${API_BASE}/transcript/${videoId}`),
        ]);

        if (notesRes.ok) {
            const notes = await notesRes.json();
            displayNotes(notes);
        }

        if (transcriptRes.ok) {
            const transcript = await transcriptRes.json();
            displayTranscript(transcript);
        }

        document.getElementById('results-section').style.display = 'block';
        // Clear Q&A
        document.getElementById('qa-messages').innerHTML = '<p class="empty-state">Ask a question about this video</p>';
    } catch (e) {
        console.error('Load notes error:', e);
    }
}

function displayNotes(data) {
    const notes = data.structured_notes || {};

    document.getElementById('notes-title').textContent = notes.title || 'Notes';
    document.getElementById('notes-summary').textContent = notes.summary || '';

    const sectionsContainer = document.getElementById('notes-sections');
    sectionsContainer.innerHTML = '';

    for (const section of notes.sections || []) {
        const div = document.createElement('div');
        div.className = 'note-section';
        div.innerHTML = `
            <h3>${escapeHtml(section.heading)}</h3>
            <ul>
                ${(section.key_points || []).map(p => `<li>${escapeHtml(p)}</li>`).join('')}
            </ul>
        `;
        sectionsContainer.appendChild(div);
    }

    // Timestamps
    const tsList = document.getElementById('timestamps-list');
    tsList.innerHTML = '';
    for (const ts of data.timestamps || []) {
        const div = document.createElement('div');
        div.className = 'timestamp-item';
        div.innerHTML = `
            <span class="time">${escapeHtml(ts.time)}</span>
            <span class="label">${escapeHtml(ts.label)}</span>
        `;
        tsList.appendChild(div);
    }

    // Action items
    const actionsList = document.getElementById('actions-list');
    actionsList.innerHTML = '';
    for (const item of data.action_items || []) {
        const div = document.createElement('div');
        div.className = 'action-item';
        const assignee = item.assignee ? ` <span class="assignee">@${escapeHtml(item.assignee)}</span>` : '';
        const ts = item.timestamp ? ` [${escapeHtml(item.timestamp)}]` : '';
        div.innerHTML = `
            <input type="checkbox" class="checkbox">
            <span class="text">${escapeHtml(item.action)}${assignee}${ts}</span>
        `;
        actionsList.appendChild(div);
    }
}

function displayTranscript(data) {
    const container = document.getElementById('transcript-content');
    container.innerHTML = '';

    for (const seg of data.segments || []) {
        const div = document.createElement('div');
        div.className = 'transcript-segment';
        const min = Math.floor(seg.start / 60);
        const sec = Math.floor(seg.start % 60);
        const time = `${String(min).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
        div.innerHTML = `
            <span class="time">${time}</span>
            <span class="text">${escapeHtml(seg.text)}</span>
        `;
        container.appendChild(div);
    }
}

// ============================================================
// Q&A
// ============================================================

async function askQuestion() {
    const input = document.getElementById('qa-question');
    const question = input.value.trim();
    if (!question || !currentVideoId) return;

    const messages = document.getElementById('qa-messages');

    // Clear empty state
    if (messages.querySelector('.empty-state')) {
        messages.innerHTML = '';
    }

    // Add user message
    const userMsg = document.createElement('div');
    userMsg.className = 'qa-message user';
    userMsg.textContent = question;
    messages.appendChild(userMsg);

    input.value = '';
    messages.scrollTop = messages.scrollHeight;

    try {
        const res = await fetch(`${API_BASE}/query/${currentVideoId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question }),
        });

        const data = await res.json();

        const asstMsg = document.createElement('div');
        asstMsg.className = 'qa-message assistant';
        asstMsg.textContent = data.answer || 'No answer available.';
        messages.appendChild(asstMsg);
    } catch (e) {
        const errMsg = document.createElement('div');
        errMsg.className = 'qa-message assistant';
        errMsg.textContent = 'Error: ' + e.message;
        messages.appendChild(errMsg);
    }

    messages.scrollTop = messages.scrollHeight;
}

// ============================================================
// Library
// ============================================================

async function refreshLibrary() {
    try {
        const res = await fetch(`${API_BASE}/library`);
        const videos = await res.json();

        const list = document.getElementById('library-list');
        if (videos.length === 0) {
            list.innerHTML = '<p class="empty-state">No videos yet</p>';
            return;
        }

        list.innerHTML = '';
        for (const v of videos) {
            const div = document.createElement('div');
            div.className = 'library-item' + (v.video_id === currentVideoId ? ' active' : '');
            div.dataset.videoId = v.video_id;
            div.dataset.title = (v.title || '').toLowerCase();

            const durationMin = (v.duration_seconds != null && !isNaN(v.duration_seconds))
                ? (v.duration_seconds / 60).toFixed(1)
                : '?';
            div.innerHTML = `
                <div class="title">${escapeHtml(v.title || v.video_id)}</div>
                <div class="meta">${durationMin} min &middot; ${v.processing_date ? v.processing_date.substring(0, 10) : ''}</div>
            `;
            div.onclick = () => loadNotes(v.video_id);
            list.appendChild(div);
        }
    } catch (e) {
        console.error('Library refresh error:', e);
    }
}

function filterLibrary() {
    const query = document.getElementById('library-search').value.toLowerCase();
    const items = document.querySelectorAll('.library-item');
    items.forEach(item => {
        const title = item.dataset.title || '';
        item.style.display = title.includes(query) ? '' : 'none';
    });
}

// ============================================================
// Tabs
// ============================================================

function switchTab(name) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));

    const btn = document.querySelector(`.tab[onclick="switchTab('${name}')"]`);
    if (btn) btn.classList.add('active');

    const content = document.getElementById(`tab-${name}`);
    if (content) content.classList.add('active');
}

// ============================================================
// Export
// ============================================================

async function exportNotes(format) {
    if (!currentVideoId) return;
    window.open(`${API_BASE}/export/${currentVideoId}?format=${format}`, '_blank');
}

// ============================================================
// Utilities
// ============================================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text || '';
    return div.innerHTML;
}

// ============================================================
// Init
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    refreshLibrary();
});
