const $ = (selector) => document.querySelector(selector);
let jobs = [];
let currentStatus = 'saved';
const statuses = ['saved', 'applied', 'interview', 'offer', 'rejected'];
const transitions = { saved: ['applied'], applied: ['interview', 'rejected'], interview: ['offer', 'rejected'], offer: [], rejected: [] };

// All server-provided content goes through textContent, never HTML interpolation.
function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function message(selector, text, error = false) {
  const node = $(selector);
  node.textContent = text || '';
  node.classList.toggle('error', error);
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Invalid request. Please check your input.');
  return body;
}
function jsonRequest(method, body) {
  return { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
}
function applyLink(url) {
  const link = element('a', 'Open posting ↗');
  try {
    const parsed = new URL(url);
    if (['http:', 'https:'].includes(parsed.protocol)) link.href = parsed.href;
  } catch (_) { /* A malformed link is displayed without navigation. */ }
  link.target = '_blank';
  link.rel = 'noopener noreferrer';
  return link;
}
function renderProfile(profile) {
  const container = $('#profile');
  container.replaceChildren();
  if (!profile) {
    container.append(element('p', 'Upload a resume to get started.', 'muted'));
    return;
  }
  container.append(element('p', `File: ${profile.filename}`, 'muted'));
  const parsed = profile.parsed;
  if (parsed) {
    container.append(element('h3', parsed.name || 'Resume profile'));
    container.append(element('p', parsed.email));
    container.append(element('p', `Total experience: ${parsed.total_years_experience} years`));
    container.append(element('h4', 'Skills'), element('p', parsed.skills.join(', ') || 'Not provided'));
    container.append(element('h4', 'Experience'));
    for (const role of parsed.roles) {
      container.append(element('p', `${role.title} · ${role.company} · ${role.years} years`));
      const list = element('ul');
      role.bullets.forEach((bullet) => list.append(element('li', bullet)));
      container.append(list);
    }
    const education = element('ul');
    parsed.education.forEach((entry) => education.append(element('li', entry)));
    container.append(element('h4', 'Education'), education);
    const details = element('details');
    details.append(element('summary', 'Extracted text'), element('pre', profile.raw_text));
    container.append(details);
  } else {
    container.append(element('h3', 'Extracted resume text'), element('pre', profile.raw_text || 'No text found in this PDF.'));
  }
}
async function loadJobs() {
  const result = await api('/jobs');
  jobs = result.jobs;
  renderJobs();
  renderTracker();
}
function statusActions(job, getNotes = () => job.notes) {
  const actions = element('div', undefined, 'actions');
  for (const status of transitions[job.status]) {
    const button = element('button', status === 'applied' ? 'Mark applied' : `Mark ${status}`);
    button.type = 'button';
    button.addEventListener('click', async () => {
      button.disabled = true;
      try {
        await api(`/jobs/${job.id}`, jsonRequest('PATCH', { status, notes: getNotes() }));
        await loadJobs();
        message('#tracker-message', `Moved to ${status}.`);
      } catch (error) { message('#tracker-message', error.message, true); button.disabled = false; }
    });
    actions.append(button);
  }
  return actions;
}
function renderJobs() {
  const body = $('#job-rows');
  body.replaceChildren();
  for (const job of jobs) {
    const row = element('tr');
    for (const value of [job.title, job.company, job.location, `${job.source} / ${job.portal}`, job.status]) {
      row.append(element('td', value));
    }
    const cell = element('td');
    cell.append(applyLink(job.job_url));
    row.append(cell);
    const actions = element('td');
    actions.append(statusActions(job));
    row.append(actions);
    body.append(row);
  }
  if (!jobs.length) {
    const row = element('tr');
    const cell = element('td', 'No jobs yet. Refresh jobs or add a posting URL.');
    cell.colSpan = 7;
    row.append(cell);
    body.append(row);
  }
}
function renderTracker() {
  const tabs = $('#status-tabs');
  tabs.replaceChildren();
  for (const status of statuses) {
    const count = jobs.filter((job) => job.status === status).length;
    const button = element('button', `${status[0].toUpperCase() + status.slice(1)} (${count})`);
    button.type = 'button';
    button.setAttribute('aria-pressed', String(currentStatus === status));
    button.addEventListener('click', () => { currentStatus = status; renderTracker(); });
    tabs.append(button);
  }
  const container = $('#tracker');
  container.replaceChildren();
  for (const job of jobs.filter((entry) => entry.status === currentStatus)) {
    const card = element('article', undefined, 'card');
    card.append(element('h3', job.title), element('p', job.company));
    card.append(element('p', `${job.portal} · ${job.location || 'Location not listed'}`, 'muted'));
    card.append(element('p', job.days_since_applied === null ? 'Not applied yet' : `${job.days_since_applied} days since applied`, 'muted'));
    card.append(applyLink(job.job_url));
    const notesForm = element('form', undefined, 'notes-form');
    const label = element('label', 'Notes');
    const notes = element('textarea');
    notes.value = job.notes;
    notes.maxLength = 10000;
    notes.placeholder = 'Referral from X…';
    label.append(notes);
    const save = element('button', 'Save notes');
    save.type = 'submit';
    notesForm.append(label, save);
    notesForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      save.disabled = true;
      try {
        await api(`/jobs/${job.id}`, jsonRequest('PATCH', { notes: notes.value }));
        job.notes = notes.value;
        message('#tracker-message', 'Notes saved.');
      } catch (error) { message('#tracker-message', error.message, true); }
      finally { save.disabled = false; }
    });
    card.append(notesForm);
    card.append(statusActions(job, () => notes.value));
    container.append(card);
  }
  if (!container.children.length) container.append(element('p', `No ${currentStatus} jobs.`, 'muted'));
}
$('#resume-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = event.currentTarget.querySelector('button');
  button.disabled = true;
  message('#resume-message', 'Reading resume…');
  try {
    const body = new FormData();
    body.append('file', $('#resume-file').files[0]);
    const result = await api('/upload', { method: 'POST', body });
    renderProfile(result.profile);
    message('#resume-message', result.warning || 'Resume parsed and saved.');
  } catch (error) { message('#resume-message', error.message, true); }
  finally { button.disabled = false; }
});
$('#search-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = event.currentTarget.querySelector('button');
  button.disabled = true;
  message('#search-message', 'Fetching Indeed, LinkedIn, and Glassdoor. Please wait…');
  try {
    const result = await api('/refresh-jobs', jsonRequest('POST', {
      search_term: $('#search-term').value, location: $('#location').value,
    }));
    message('#search-message', result.sources.map((source) =>
      `${source.site}: ${source.added} new, ${source.updated} refreshed, ${source.skipped} skipped.${source.warning ? ` ${source.warning}` : ''}`
    ).join('\n'));
    await loadJobs();
  } catch (error) { message('#search-message', error.message, true); }
  finally { button.disabled = false; }
});
$('#add-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = event.currentTarget.querySelector('button');
  button.disabled = true;
  message('#add-message', 'Fetching posting…');
  try {
    const result = await api('/add-job', jsonRequest('POST', { url: $('#job-url').value }));
    message('#add-message', `${result.created ? 'Job added.' : 'Existing job refreshed.'}${result.warning ? ` ${result.warning}` : ''}`);
    $('#job-url').value = '';
    await loadJobs();
  } catch (error) { message('#add-message', error.message, true); }
  finally { button.disabled = false; }
});
async function initialize() {
  try { renderProfile((await api('/profile')).profile); }
  catch (error) { message('#resume-message', error.message, true); }
  try { await loadJobs(); }
  catch (error) { message('#tracker-message', error.message, true); }
}
initialize();
