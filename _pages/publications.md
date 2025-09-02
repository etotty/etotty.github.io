---
layout: page
permalink: /publications/
title: Publications
description: Publications by type or by topic, sorted by year
nav: true
nav_order: 2
---

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
  const list = document.getElementById("pub-list");
  const items = Array.from(list.querySelectorAll(".pub"));

  // Remove old headers
  Array.from(list.querySelectorAll(".pub-group-header")).forEach(h => h.remove());

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

    // Insert header before the first item of the group
    const header = document.createElement("div");
    header.className = "pub-group-header";
    header.textContent = key;
    list.insertBefore(header, group[0]);

    // Show all items, sorted by year descending
    group.sort((a, b) => (parseInt(b.dataset.year) || 0) - (parseInt(a.dataset.year) || 0)).forEach(pub => {
      pub.style.display = "flex";  // Ensure visible
    });
  });

  // Re-initialize Bootstrap popovers
  if (typeof $ !== "undefined" && $.fn.popover) {
    $('[data-toggle="popover"]').popover();
  }
}

// Default view on page load
document.addEventListener("DOMContentLoaded", () => filterPubs('type'));
</script>

