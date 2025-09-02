---
layout: page
permalink: /publications/
title: publications
description: publications by type or by topic
nav: true
nav_order: 2
---

<!-- Bibsearch Feature -->
{% include bib_search.liquid %}

<!-- Toolbar for grouping options -->
<div id="pub-toolbar" style="margin-bottom: 1em;">
  <button onclick="groupPubs('type')">Group by Type</button>
  <button onclick="groupPubs('topic')">Group by Topic</button>
</div>

<!-- Publications list -->
<div class="publications" id="pub-list">
  {% bibliography %}
</div>

<script>
function groupPubs(mode) {
  const list = document.getElementById("pub-list");
  const items = Array.from(list.querySelectorAll(".pub"));

  // clear the container
  list.innerHTML = "";

  // Group publications
  const groups = {};
  items.forEach(item => {
    const year = parseInt(item.dataset.year) || 0;
    const type = item.dataset.type || "Unspecified";
    const topic = item.dataset.topic || "Unspecified";

    let key;
    if (mode === "type") {
      key = type;
    } else if (mode === "topic") {
      key = topic;
    }

    if (!groups[key]) groups[key] = [];
    groups[key].push({ element: item, year: year });
  });

  // Render groups sorted by year descending
  const sortedKeys = Object.keys(groups).sort();

  sortedKeys.forEach(key => {
    const header = document.createElement("h3");
    header.textContent = key;
    list.appendChild(header);

    // Sort publications by year descending within the group
    groups[key].sort((a, b) => b.year - a.year).forEach(pub => {
      list.appendChild(pub.element);
    });
  });
}

// Default grouping on page load
document.addEventListener("DOMContentLoaded", () => {
  groupPubs("type");
});
</script>
