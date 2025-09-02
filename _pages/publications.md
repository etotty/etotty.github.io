---
layout: page
permalink: /publications/
title: publications
description: publications by type (default) or by topic
nav: true
nav_order: 2
---

{% include bib_search.liquid %}

<!-- Toolbar for grouping -->
<div id="pub-toolbar" style="margin-bottom: 1em;">
  <button onclick="groupPubs('type')">Group by Type</button>
  <button onclick="groupPubs('topic')">Group by Topic</button>
</div>

<!-- Container for publications -->
<div class="publications" id="pub-list">
  {% bibliography %}
</div>

<style>
.pub-group-header {
  width: 100%;
  margin-top: 2em;
  margin-bottom: 0.5em;
  font-size: 1.5em;
  font-weight: bold;
}
.pub {
  display: flex !important;
  flex-wrap: wrap;
}
</style>

<script>
function groupPubs(mode) {
  const container = document.getElementById("pub-list");
  const pubs = Array.from(container.querySelectorAll(".pub"));

  if (pubs.length === 0) return;

  // Remove existing headers
  Array.from(container.querySelectorAll(".pub-group-header")).forEach(h => h.remove());

  // Group pubs by type or topic
  const groups = {};
  pubs.forEach(pub => {
    const key = mode === 'type' ? (pub.dataset.type || 'Unspecified') : (pub.dataset.topic || 'Unspecified');
    const year = parseInt(pub.dataset.year) || 0;
    if (!groups[key]) groups[key] = [];
    groups[key].push({ element: pub, year });
  });

  // Clear container
  container.innerHTML = '';

  // Sort groups alphabetically
  Object.keys(groups).sort().forEach(key => {
    const header = document.createElement("div");
    header.textContent = key;
    header.className = "pub-group-header";
    container.appendChild(header);

    // Sort publications in each group by year descending
    groups[key].sort((a, b) => b.year - a.year).forEach(pub => {
      container.appendChild(pub.element);
      pub.element.style.display = "flex";
    });
  });

  // Re-initialize popovers or other JS features if needed
  if (typeof $ !== "undefined" && $.fn.popover) {
    $('[data-toggle="popover"]').popover();
  }
}

// Wait until the bibliography is rendered before grouping
function initGrouping() {
  const container = document.getElementById("pub-list");
  if (container.querySelectorAll(".pub").length === 0) {
    setTimeout(initGrouping, 100); // check again
  } else {
    groupPubs('type'); // default
  }
}
document.addEventListener("DOMContentLoaded", initGrouping);
</script>


