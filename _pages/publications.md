---
layout: page
permalink: /publications/
title: publications
description: publications by type or by topic
nav: true
nav_order: 2
---

<!-- _pages/publications.md -->

<!-- Bibsearch Feature -->
{% include bib_search.liquid %}

<!-- Toolbar for grouping options -->
<div id="pub-toolbar" style="margin-bottom: 1em;">
  <button onclick="groupPubs('type-year')">Group by Type–Year</button>
  <button onclick="groupPubs('topic-year')">Group by Topic–Year</button>
</div>

<!-- Publications list -->
<div class="publications">
  <ul id="pub-list">
    {% bibliography %}
  </ul>
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
    const year = item.dataset.year;
    const type = item.dataset.type;
    const topic = item.dataset.topic;

    let key;
    if (mode === "type-year") {
      key = type + " – " + year;
    } else if (mode === "topic-year") {
      key = topic + " – " + year;
    }
    if (!groups[key]) groups[key] = [];
    groups[key].push(item);
  });

  // Sort groups (by year descending inside type/topic)
  const sortedKeys = Object.keys(groups).sort((a, b) => {
    const yearA = parseInt(a.split("–").pop());
    const yearB = parseInt(b.split("–").pop());
    return yearB - yearA; // descending
  });

  // Render grouped list
  sortedKeys.forEach(key => {
    const header = document.createElement("h3");
    header.textContent = key;
    list.appendChild(header);

    groups[key].forEach(pub => {
      list.appendChild(pub);
    });
  });
}

// Default grouping on page load
document.addEventListener("DOMContentLoaded", () => {
  groupPubs("type-year");
});
</script>
