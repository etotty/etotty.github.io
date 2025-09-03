---
layout: page
permalink: /publications_factormodel/
title: publications
description: publications by categories in reversed chronological order
nav: false
nav_order: 2
---

<!-- Navigation buttons -->
<ul class="nav nav-pills mb-4">
  <li class="nav-item">
    <a class="nav-link" href="{{ '/publications/' | relative_url }}">All</a>
  </li>
  <li class="nav-item">
    <a class="nav-link" href="{{ '/publications_dataprivacy/' | relative_url }}">Data Privacy</a>
  </li>
  <li class="nav-item">
    <a class="nav-link" href="{{ '/publications_education/' | relative_url }}">Education</a>
  </li>
  <li class="nav-item">
    <a class="nav-link active" href="{{ '/publications_factormodel/' | relative_url }}">Factor Model</a>
  </li>
  <li class="nav-item">
    <a class="nav-link" href="{{ '/publications_minimumwage/' | relative_url }}">Minimum Wage</a>
  </li>
  <li class="nav-item">
    <a class="nav-link" href="{{ '/publications_other/' | relative_url }}">Other</a>
  </li>
</ul>

<div class="publications">
 {% bibliography --query @*[topic~=factor model] --group_by type %}
</div>
