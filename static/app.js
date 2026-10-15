const $ = (selector) => document.querySelector(selector);
let jobs = [];
let currentStatus = 'saved';
let aiEnabled = false;
let aiEnableMessage = 'Add OPENAI_API_KEY to .env to enable';
let discoveryProfileVersion = null;
let discoveryBusy = false;
let currentProfile = null;
let showHiddenAuth = false;
const authorizationLevels = { 'US Citizen': 4, 'Green Card': 3, 'H1B or other work visa': 2, 'F-1 STEM OPT': 1, 'F-1 OPT': 1, 'Need sponsorship': 0 };
const requiredLevels = { citizen: 4, permanent: 3, authorized: 1, sponsors: 0, unknown: 0 };
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
  currentProfile = profile;
  $('#work-authorization').value = profile?.work_authorization || 'F-1 STEM OPT';
  const years = profile?.parsed?.total_years_experience;
  for (const checkbox of document.querySelectorAll('#experience-filters input')) {
    checkbox.checked = checkbox.value === 'unspecified' || typeof years !== 'number' || !Number.isFinite(years) || years < 0 || checkbox.value === experienceBucket(years);
  }
  const container = $('#profile');
  container.replaceChildren();
  if (!profile || !profile.filename) {
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
  aiEnabled = result.ai_enabled;
  aiEnableMessage = result.ai_message;
  renderJobs();
  renderTracker();
}
function rememberLocation() {
  const location = $('#location').value.trim() || 'USA';
  $('#location').value = location;
  try { localStorage.setItem('jobmatch.location', location); } catch (_) { /* Storage can be disabled. */ }
  return location;
}
function sourceSummary(sources) {
  return sources.map((source) => `${source.site}: ${source.added} new, ${source.updated} refreshed, ${source.skipped} skipped.${source.date_filtered ? ` ${source.date_filtered} outside date range.` : ''}${source.auth_hidden ? ` ${source.auth_hidden} hidden by work authorization.` : ''}${source.experience_filtered ? ` ${source.experience_filtered} outside experience buckets.` : ''}${source.warning ? ` ${source.warning}` : ''}`).join('\n');
}
function updateDiscoveryControls() {
  for (const selector of ['#discover-jobs', '#refresh-jobs', '#location', '#search-term', '#date-range']) $(selector).disabled = discoveryBusy;
  for (const checkbox of document.querySelectorAll('#query-options input')) checkbox.disabled = discoveryBusy;
  $('#run-discovery').disabled = discoveryBusy || !document.querySelector('#query-options input:checked');
}
function searchFilters() {
  return { hours_old: Number($('#date-range').value), show_hidden_auth: showHiddenAuth,
    experience_buckets: [...document.querySelectorAll('#experience-filters input:checked')].map((input) => input.value) };
}
function postedTime(value) {
  if (!value) return NaN;
  const text = String(value).trim();
  if (/^\d{4}-\d{2}-\d{2}$/.test(text)) {
    const date = new Date(`${text}T23:59:59.999Z`);
    return Number.isNaN(date.getTime()) || date.toISOString().slice(0, 10) !== text ? NaN : date.getTime();
  }
  const iso = text.replace(' ', 'T');
  return Date.parse(/(?:Z|[+-]\d{2}:?\d{2})$/i.test(iso) ? iso : `${iso}Z`);
}
function authorizationAllowed(job) {
  return (requiredLevels[job.auth_tier] || 0) <= authorizationLevels[$('#work-authorization').value];
}
function experienceBucket(years) {
  if (years === null || years === undefined) return 'unspecified';
  if (years < 3) return '0-3';
  if (years < 6) return '3-6';
  if (years < 10) return '6-10';
  return '10+';
}
function visibleJobs(applyDateRange = true, includeAuthHidden = showHiddenAuth) {
  const filters = searchFilters();
  const cutoff = Date.now() - filters.hours_old * 60 * 60 * 1000;
  return jobs.filter((job) => {
    if (!includeAuthHidden && !authorizationAllowed(job)) return false;
    if (applyDateRange && !filters.experience_buckets.includes(experienceBucket(job.exp_min))) return false;
    const posted = postedTime(job.posted_at);
    return !applyDateRange || Number.isNaN(posted) || posted >= cutoff;
  });
}
function authBadge(job) {
  return element('span', job.auth_tier, `visa-badge ${job.auth_tier}`);
}
function saveFilters() {
  try {
    localStorage.setItem('jobmatch.hours_old', $('#date-range').value);
  } catch (_) { /* Use the current controls when storage is unavailable. */ }
  renderJobs();
  renderTracker();
}
function alsoOn(job) {
  const names = [...new Set(job.also_on.map((entry) => entry.source))];
  return names.length ? element('p', `also on ${names.join(', ')}`, 'muted also-on') : null;
}
function scoreBadge(job) {
  return element('span', job.match_score === null ? 'Unscored' : `${job.match_score.toFixed(1)}%`,
    job.match_score === null ? 'score unscored' : 'score');
}
function skillChips(job) {
  const container = element('div', undefined, 'skill-chips');
  for (const [kind, skills] of Object.entries(job.skill_overlap)) {
    if (!skills.length) continue;
    const group = element('div', undefined, `skill-group ${kind}`);
    group.append(element('span', `${kind}: `, 'skill-label'));
    for (const skill of skills) group.append(element('span', skill, 'chip'));
    container.append(group);
  }
  return container;
}
function renderAIOutputs(job, container, changedKind = null) {
  const existingLetter = container.querySelector('[data-kind="cover_letter"]');
  container.replaceChildren();
  const labels = { fit_report: 'Fit report', tailor_bullets: 'Tailored bullets', cover_letter: 'Cover letter' };
  for (const [kind, label] of Object.entries(labels)) {
    const output = job.ai[kind];
    if (!output) continue;
    if (kind === 'cover_letter' && existingLetter && changedKind && changedKind !== 'cover_letter') {
      // Generating another kit item must not discard an unsaved letter edit.
      container.append(existingLetter);
      continue;
    }
    const section = element('div', undefined, 'ai-output');
    section.dataset.kind = kind;
    section.append(element('h4', label));
    if (kind === 'tailor_bullets') {
      const list = element('ul');
      output.content.forEach((bullet) => list.append(element('li', bullet)));
      section.append(list);
    } else if (kind === 'cover_letter') {
      const draftLabel = element('label', `Cover letter draft for ${job.company}`);
      const draft = element('textarea');
      draft.value = output.content;
      draft.maxLength = 20000;
      draft.rows = 12;
      draftLabel.append(draft);
      section.append(draftLabel);
      const actions = element('div', undefined, 'actions');
      const save = element('button', 'Save edits');
      const copy = element('button', 'Copy');
      save.type = copy.type = 'button';
      const status = element('p', undefined, 'message');
      status.setAttribute('role', 'status');
      save.addEventListener('click', async () => {
        save.disabled = true;
        status.textContent = '';
        try {
          const result = await api(`/jobs/${job.id}/ai/cover_letter`, jsonRequest('PATCH', { content: draft.value }));
          output.content = result.content;
          status.textContent = 'Edits saved.';
          status.classList.remove('error');
        } catch (error) { status.textContent = error.message; status.classList.add('error'); }
        finally { save.disabled = false; }
      });
      copy.addEventListener('click', async () => {
        try {
          if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(draft.value);
          else {
            draft.focus();
            draft.select();
            if (!document.execCommand('copy')) throw new Error('Select the letter and copy it manually.');
          }
          status.textContent = 'Copied.';
          status.classList.remove('error');
        } catch (_) { status.textContent = 'Select the letter and copy it manually.'; status.classList.add('error'); }
      });
      actions.append(save, copy);
      section.append(actions, status);
    } else {
      section.append(element('p', output.content, 'ai-text'));
    }
    container.append(section);
  }
}
function fitKit(job) {
  const details = element('details', undefined, 'fit-kit');
  details.append(element('summary', 'AI fit kit'));
  if (!aiEnabled) details.append(element('p', aiEnableMessage, 'muted'));
  const actions = element('div', undefined, 'actions');
  const status = element('p', undefined, 'message');
  status.setAttribute('role', 'status');
  const outputs = element('div');
  const buttons = [];
  for (const [kind, label] of Object.entries({ fit_report: 'Fit report', tailor_bullets: 'Tailor bullets', cover_letter: 'Cover letter' })) {
    const button = element('button', job.ai[kind] ? `Regenerate ${label.toLowerCase()}` : label);
    button.type = 'button';
    button.disabled = !aiEnabled;
    if (!aiEnabled) button.title = aiEnableMessage;
    buttons.push(button);
    button.addEventListener('click', async () => {
      buttons.forEach((entry) => { entry.disabled = true; });
      status.textContent = `Generating ${label.toLowerCase()}…`;
      status.classList.remove('error');
      try {
        const result = await api(`/jobs/${job.id}/ai/${kind}`, { method: 'POST' });
        if (!result.enabled) {
          aiEnabled = false;
          buttons.forEach((entry) => { entry.title = result.message; });
          status.textContent = result.message;
        } else {
          job.ai[kind] = { content: result.content, created_at: result.created_at };
          renderAIOutputs(job, outputs, kind);
          button.textContent = `Regenerate ${label.toLowerCase()}`;
          status.textContent = `${label} saved.`;
        }
      } catch (error) { status.textContent = error.message; status.classList.add('error'); }
      finally { buttons.forEach((entry) => { entry.disabled = !aiEnabled; }); }
    });
    actions.append(button);
  }
  renderAIOutputs(job, outputs);
  details.append(actions, status, outputs);
  return details;
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
  const visible = visibleJobs();
  const authHiddenCount = visibleJobs(true, true).filter((job) => !authorizationAllowed(job)).length;
  const reveal = $('#show-auth-hidden');
  reveal.hidden = authHiddenCount === 0;
  reveal.textContent = showHiddenAuth ? `Hide ${authHiddenCount} restricted` : `Show ${authHiddenCount} hidden`;
  reveal.setAttribute('aria-expanded', String(showHiddenAuth));
  $('#filter-message').textContent = `${visible.length} of ${jobs.length} saved jobs shown. Date range filters results; older applications stay in the tracker.`;
  for (const job of visible) {
    const row = element('tr');
    row.dataset.jobId = job.id;
    const title = element('td', job.title, 'job-title');
    title.append(element('p', job.exp_min === null ? 'Experience unspecified' : `Minimum experience: ${job.exp_min} years`, 'muted'));
    title.append(skillChips(job));
    const score = element('td');
    score.append(scoreBadge(job));
    row.append(title, score);
    const visa = element('td');
    visa.append(authBadge(job));
    row.append(visa);
    for (const value of [job.company, job.location]) {
      row.append(element('td', value));
    }
    const source = element('td', `${job.source} / ${job.portal}`);
    const hint = alsoOn(job);
    if (hint) source.append(hint);
    row.append(source, element('td', job.status));
    const cell = element('td');
    cell.append(applyLink(job.job_url));
    row.append(cell);
    const actions = element('td');
    actions.append(statusActions(job));
    row.append(actions);
    const ai = element('td');
    ai.append(fitKit(job));
    row.append(ai);
    body.append(row);
  }
  if (!visible.length) {
    const row = element('tr');
    const cell = element('td', jobs.length ? 'No jobs match these filters. Adjust experience or work authorization, or show hidden roles.' : 'No jobs yet. Refresh jobs or add a posting URL.');
    cell.colSpan = 10;
    row.append(cell);
    body.append(row);
  }
}
function renderTracker() {
  const tabs = $('#status-tabs');
  tabs.replaceChildren();
  const trackerJobs = visibleJobs(false);
  for (const status of statuses) {
    const count = trackerJobs.filter((job) => job.status === status).length;
    const button = element('button', `${status[0].toUpperCase() + status.slice(1)} (${count})`);
    button.type = 'button';
    button.setAttribute('aria-pressed', String(currentStatus === status));
    button.addEventListener('click', () => { currentStatus = status; renderTracker(); });
    tabs.append(button);
  }
  const container = $('#tracker');
  container.replaceChildren();
  for (const job of trackerJobs.filter((entry) => entry.status === currentStatus)) {
    const card = element('article', undefined, 'card');
    card.append(element('h3', job.title), element('p', job.company));
    card.append(scoreBadge(job), authBadge(job), skillChips(job));
    card.append(element('p', `${job.portal} · ${job.location || 'Location not listed'}`, 'muted'));
    const hint = alsoOn(job);
    if (hint) card.append(hint);
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
    $('#discovery-plan').hidden = true;
    discoveryProfileVersion = null;
    message('#discovery-message', '');
    message('#resume-message', result.warning || 'Resume parsed and saved.');
    await loadJobs();
  } catch (error) { message('#resume-message', error.message, true); }
  finally { button.disabled = false; }
});
$('#search-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  discoveryBusy = true;
  updateDiscoveryControls();
  rememberLocation();
  message('#search-message', 'Generating job-search queries from your resume…');
  message('#discovery-message', '');
  $('#discovery-plan').hidden = true;
  try {
    const result = await api('/discovery/queries', jsonRequest('POST', { role_override: $('#search-term').value }));
    discoveryProfileVersion = result.profile_uploaded_at;
    $('#detected-title').textContent = result.detected_titles.length
      ? `Detected title${result.detected_titles.length > 1 ? 's' : ''}: ${result.detected_titles.join(', ')} (${result.title_source}).`
      : 'No recent title detected. Queries use your skills; enter a role override to target a title.';
    $('#discovery-skills').textContent = `Resume skills: ${result.skills.join(', ') || 'None detected'}. Query builder: ${result.method}.`;
    const options = $('#query-options');
    options.replaceChildren();
    for (const query of result.queries) {
      const label = element('label', undefined, 'query-option');
      const checkbox = element('input');
      checkbox.type = 'checkbox';
      checkbox.value = query;
      checkbox.checked = true;
      checkbox.addEventListener('change', updateDiscoveryControls);
      label.append(checkbox, element('span', query));
      options.append(label);
    }
    $('#discovery-plan').hidden = false;
    message('#search-message', `${result.queries.length} queries generated. Uncheck any you do not want, then Run discovery.${result.warning ? ` ${result.warning}` : ''}`);
  } catch (error) { message('#search-message', error.message, true); }
  finally { discoveryBusy = false; updateDiscoveryControls(); }
});
$('#run-discovery').addEventListener('click', async () => {
  const queries = [...document.querySelectorAll('#query-options input:checked')].map((input) => input.value);
  if (!queries.length || !discoveryProfileVersion) return;
  const version = discoveryProfileVersion;
  const location = rememberLocation();
  const filters = searchFilters();
  discoveryBusy = true;
  updateDiscoveryControls();
  const lines = [];
  let totalNew = 0;
  try {
    for (const [index, query] of queries.entries()) {
      message('#discovery-message', [`Running ${index + 1}/${queries.length}: ${query} — ${location}`, ...lines].join('\n'));
      const result = await api('/discovery/run', jsonRequest('POST', {
        queries: [query], location, profile_uploaded_at: version, ...filters,
      }));
      totalNew += result.new_jobs;
      for (const entry of result.queries) {
        lines.push(`${entry.query}: ${entry.jobs_found} unique jobs found.\n${sourceSummary(entry.sources)}`);
      }
      lines.push(`Auto-scored ${result.scoring.scored} new jobs.${result.warning ? ` ${result.warning}` : ''}`);
      await loadJobs();
    }
    message('#discovery-message', [`Discovery complete: ${queries.length} queries, ${totalNew} new jobs saved.`, ...lines].join('\n'));
  } catch (error) { message('#discovery-message', [...lines, error.message].join('\n'), true); }
  finally { discoveryBusy = false; updateDiscoveryControls(); }
});
$('#refresh-jobs').addEventListener('click', async () => {
  const role = $('#search-term').value.trim();
  if (!role) {
    message('#search-message', 'Enter a role override for manual Refresh jobs, or use Discover jobs for me.');
    return;
  }
  discoveryBusy = true;
  updateDiscoveryControls();
  message('#search-message', 'Fetching Indeed, LinkedIn, and Glassdoor. Please wait…');
  try {
    const result = await api('/refresh-jobs', jsonRequest('POST', {
      search_term: role, location: rememberLocation(), ...searchFilters(),
    }));
    message('#search-message', sourceSummary(result.sources));
    await loadJobs();
  } catch (error) { message('#search-message', error.message, true); }
  finally { discoveryBusy = false; updateDiscoveryControls(); }
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
$('#score-jobs').addEventListener('click', async (event) => {
  const button = event.currentTarget;
  button.disabled = true;
  message('#score-message', 'Scoring locally on CPU… First use downloads all-MiniLM-L6-v2; this may take a few minutes.');
  try {
    const result = await api('/score-jobs', { method: 'POST' });
    await loadJobs();
    message('#score-message', `${result.scored} jobs scored. ${result.embedded} new embeddings; ${result.cached} cached embeddings reused.`);
  } catch (error) { message('#score-message', error.message, true); }
  finally { button.disabled = false; }
});
$('#work-authorization').addEventListener('change', async (event) => {
  const selector = event.currentTarget;
  const previous = currentProfile?.work_authorization || 'F-1 STEM OPT';
  showHiddenAuth = false;
  renderJobs();
  renderTracker();
  selector.disabled = true;
  message('#authorization-message', 'Saving work authorization…');
  try {
    const result = await api('/profile', jsonRequest('PATCH', { work_authorization: selector.value }));
    currentProfile = result.profile;
    message('#authorization-message', 'Work authorization saved. Filters updated.');
  } catch (error) {
    selector.value = previous;
    renderJobs();
    renderTracker();
    message('#authorization-message', error.message, true);
  } finally { selector.disabled = false; }
});
$('#show-auth-hidden').addEventListener('click', () => {
  showHiddenAuth = !showHiddenAuth;
  renderJobs();
  renderTracker();
});
for (const checkbox of document.querySelectorAll('#experience-filters input')) {
  checkbox.addEventListener('change', () => { renderJobs(); renderTracker(); });
}
async function initialize() {
  try { $('#location').value = localStorage.getItem('jobmatch.location') || 'USA'; } catch (_) { /* Use USA by default. */ }
  $('#location').addEventListener('change', rememberLocation);
  try {
    const hours = localStorage.getItem('jobmatch.hours_old');
    if (['24', '48', '72', '168', '720', '2160'].includes(hours)) $('#date-range').value = hours;
  } catch (_) { /* Default: 3 days. */ }
  $('#date-range').addEventListener('change', saveFilters);
  try { renderProfile((await api('/profile')).profile); }
  catch (error) { message('#resume-message', error.message, true); }
  try { await loadJobs(); }
  catch (error) { message('#tracker-message', error.message, true); }
}
initialize();
