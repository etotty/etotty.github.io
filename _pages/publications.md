---
layout: page
permalink: /publications/
title: Publications
description: Publications grouped by type or topic
nav: true
nav_order: 2
---

<!-- Bibsearch Feature -->
{% include bib_search.liquid %}

<!-- Toolbar for grouping options -->
<div id="pub-toolbar" style="margin-bottom: 1em;">
  <button onclick="filterPubs('type')">Group by Type</button>
  <button onclick="filterPubs('topic')">Group by Topic</button>
</div>

<!-- Publications list -->
<div class="publications" id="pub-list">
  {% bibliography %}
</div>

<style>
/* Ensure each publication row displays correctly */
.pub {
  display: flex !important;
  flex-wrap: wrap;
  margin-bottom: 1rem;
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
function filterPubs(mode) {
  const container = document.getElementById("pub-list");
  const items = Array.from(container.querySelectorAll(".pub"));

  // Remove old headers
  Array.from(container.querySelectorAll(".pub-group-header")).forEach(h => h.remove());

  // Build groups
  const groups = {};
  items.forEach(item => {
    const key = (mode === "type") ? (item.dataset.type || "Unspecified") : (item.dataset.topic || "Unspecified");
    if (!groups[key]) groups[key] = [];
    groups[key].push(item);
  });

  // Sort keys alphabetically
  Object.keys(groups).sort().forEach(key => {
    const group = groups[key];

    // Sort publications by year descending
    group.sort((a, b) => (parseInt(b.dataset.year) || 0) - (parseInt(a.dataset.year) || 0));

    // Show items in order and insert group header
    group.forEach((pub, i) => {
      pub.style.display = "flex";
      if (i === 0) {
        const header = document.createElement("div");
        header.className = "pub-group-header";
        header.textContent = key;
        container.insertBefore(header, pub);
      }
    });
  });

  // Re-initialize popovers
  if (typeof $ !== "undefined" && $.fn.popover) {
    $('[data-toggle="popover"]').popover();
  }
}

// Default grouping on page load
document.addEventListener("DOMContentLoaded", () => filterPubs('type'));
</script>

