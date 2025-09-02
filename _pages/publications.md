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

<style>
/* Optional: ensure pub divs are flex containers for proper Bootstrap layout */
.pub {
  display: flex !important;
  flex-wrap: wrap;
}

/* Header style for each group */
.pub-group-header {
  width: 100%;
  margin-top: 2em;
  margin-bottom: 0.5em;
  font-size: 1.5em;
  font-weight: bold;
}
</style>

<script>
function groupPubs(mode) {
  const list = document.getElementById("pub-list");
  const items = Array.from(list.querySelectorAll(".pub"));

  // Clear previous headers
  Array.from(list.querySelectorAll(".pub-group-header")).forEach(h => h.remove());

  // Group publications
  const groups = {};
  items.forEach(item => {
    const year = parseInt(item.dataset.year) || 0;
    const type = item.dataset.type || "Unspecified";
    const topic = item.dataset.topic || "Unspecified";

    const key = (mode === "type") ? type : topic;

    if (!groups[key]) groups[key] = [];
    groups[key].push({ element: item, year: year });
  });

  // Sort group keys alphabetically
  const sortedKeys = Object.keys(groups).sort();

  sortedKeys.forEach(key => {
    const group = groups[key];

    // Insert group header
    const header = document.createElement("div");
    header.textContent = key;
    header.className = "pub-group-header";
    list.appendChild(header);

    // Sort publications by year descending and set CSS order
    group.sort((a, b) => b.year - a.year).forEach((pub, index) => {
      pub.element.style.order = index;
      list.appendChild(pub.element);
      pub.element.style.display = "flex"; // ensure visible
    });
  });

  // Re-initialize Bootstrap popovers (if used)
  if (typeof $ !== "undefined" && $.fn.popover) {
    $('[data-toggle="popover"]').popover();
  }
}

// Default grouping on page load
document.addEventListener("DOMContentLoaded", () => {
  groupPubs("type");
});
</script>
