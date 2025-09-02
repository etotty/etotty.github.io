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
  <button>Group by Type</button>
  <button>Group by Topic</button>
</div>

<!-- Publications list -->
<div class="publications" id="pub-list">
  {% bibliography %}
</div>

<style>
/* Ensure .pub divs behave like flex containers for Bootstrap */
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
  const container = document.getElementById("pub-list");
  const items = Array.from(container.querySelectorAll(".pub"));

  if (!items.length) return;

  // Remove existing headers
  Array.from(container.querySelectorAll(".pub-group-header")).forEach(h => h.remove());

  // Build groups
  const groups = {};
  items.forEach(item => {
    const key = mode === "type" ? (item.dataset.type || "Unspecified") : (item.dataset.topic || "Unspecified");
    if (!groups[key]) groups[key] = [];
    groups[key].push(item);
  });

  // Sort groups alphabetically
  Object.keys(groups).sort().forEach(key => {
    const group = groups[key];

    // Sort by year descending
    group.sort((a, b) => (parseInt(b.dataset.year) || 0) - (parseInt(a.dataset.year) || 0));

    // Insert group header above first item
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

  // Re-initialize popovers (Bootstrap)
  if (typeof $ !== "undefined" && $.fn.popover) {
    $('[data-toggle="popover"]').popover();
  }
}

// Wait for DOM + Scholar entries
document.addEventListener("DOMContentLoaded", () => {
  // Delay to ensure Jekyll Scholar rendered all .pub entries
  setTimeout(() => groupPubs('type'), 100);

  // Add click handlers
  const buttons = document.querySelectorAll("#pub-toolbar button");
  if (buttons.length >= 2) {
    buttons[0].addEventListener("click", () => groupPubs('type'));
    buttons[1].addEventListener("click", () => groupPubs('topic'));
  }
});
</script>

