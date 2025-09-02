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

  // Clear the container
  list.innerHTML = "";

  // Group publications
  const groups = {};
  items.forEach(item => {
    const year = parseInt(item.dataset.year) || 0;
    const type = item.dataset.type || "Unspecified";
    const topic = item.dataset.topic || "Unspecified";

    let key = (mode === "type") ? type : topic;

    if (!groups[key]) groups[key] = [];
    groups[key].push({ element: item, year: year });
  });

  // Sort group keys alphabetically
  const sortedKeys = Object.keys(groups).sort();

  sortedKeys.forEach(key => {
    // Create a container for this group
    const groupWrapper = document.createElement("div");
    groupWrapper.className = "pub-group";

    // Add a header
    const header = document.createElement("h3");
    header.textContent = key;
    groupWrapper.appendChild(header);

    // Sort publications in this group by year descending
    groups[key].sort((a, b) => b.year - a.year).forEach(pub => {
      groupWrapper.appendChild(pub.element);
    });

    list.appendChild(groupWrapper);
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

// Default grouping on page load
document.addEventListener("DOMContentLoaded", () => {
  groupPubs("type");
});
</script>
