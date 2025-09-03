---
layout: page
permalink: /publications/
title: publications
description: publications by categories in reversed chronological order
nav: true
nav_order: 2
---

<!-- Navigation buttons -->
<ul class="nav nav-pills mb-4">
  <li class="nav-item">
    <a class="nav-link active" href="{{ '/publications/' | relative_url }}">All</a>
  </li>
</ul>

<div class="publications">
 {% bibliography --group_by type %}
</div>
