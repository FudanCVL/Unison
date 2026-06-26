/* ============================================================
   Unison — Project Page interactions
   ============================================================ */
(function () {
  "use strict";

  /* ---- Nav: solidify on scroll + back-to-top visibility ---- */
  var nav = document.getElementById("nav");
  var toTop = document.getElementById("toTop");

  function onScroll() {
    var y = window.scrollY || window.pageYOffset;
    if (nav) nav.classList.toggle("is-scrolled", y > 24);
    if (toTop) toTop.classList.toggle("is-shown", y > 600);
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  if (toTop) {
    toTop.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  /* ---- Reveal on scroll ---- */
  var reveals = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) {
            e.target.classList.add("is-visible");
            io.unobserve(e.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -8% 0px" }
    );
    reveals.forEach(function (el) { io.observe(el); });
  } else {
    reveals.forEach(function (el) { el.classList.add("is-visible"); });
  }

  /* ---- Leaderboard tabs ---- */
  var tabs = document.querySelectorAll(".lb-tab");
  tabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      var targetId = tab.getAttribute("data-target");
      tabs.forEach(function (t) { t.classList.remove("is-active"); });
      tab.classList.add("is-active");
      document.querySelectorAll(".lb-panel").forEach(function (p) {
        p.classList.toggle("is-active", p.id === targetId);
      });
    });
  });

  /* ---- Overall-column heatmap (per table) ---- */
  document.querySelectorAll("table.lb").forEach(function (table) {
    var cells = Array.prototype.slice.call(
      table.querySelectorAll("tbody .col-overall")
    );
    var vals = cells.map(function (c) {
      var n = parseFloat(c.textContent.replace(/[^0-9.]/g, ""));
      return isNaN(n) ? null : n;
    });
    var max = Math.max.apply(null, vals.filter(function (v) { return v !== null; }));
    var min = Math.min.apply(null, vals.filter(function (v) { return v !== null; }));
    cells.forEach(function (c, i) {
      var v = vals[i];
      if (v === null) return;
      var t = max === min ? 1 : (v - min) / (max - min); // 0..1
      var a = 0.07 + t * 0.26; // alpha 0.07 .. 0.33
      c.style.background =
        "linear-gradient(120deg, rgba(79,159,230," + a + "), rgba(231,131,143," + a + "))";
    });
  });

  /* ---- Copy BibTeX ---- */
  var copyBtn = document.getElementById("copyBtn");
  var bib = document.getElementById("bibtex");
  if (copyBtn && bib) {
    var label = copyBtn.querySelector("span");
    var original = label ? label.textContent : "Copy";
    copyBtn.addEventListener("click", function () {
      var text = bib.textContent;
      var done = function () {
        copyBtn.classList.add("is-copied");
        if (label) label.textContent = "Copied!";
        setTimeout(function () {
          copyBtn.classList.remove("is-copied");
          if (label) label.textContent = original;
        }, 1800);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done).catch(fallback);
      } else {
        fallback();
      }
      function fallback() {
        var ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand("copy"); done(); } catch (e) {}
        document.body.removeChild(ta);
      }
    });
  }

  /* ---- Smooth-scroll for in-page anchors (with nav offset) ---- */
  document.querySelectorAll('a[href^="#"]').forEach(function (a) {
    a.addEventListener("click", function (e) {
      var id = a.getAttribute("href");
      if (id.length < 2) return;
      var el = document.querySelector(id);
      if (!el) return;
      e.preventDefault();
      var top = el.getBoundingClientRect().top + window.pageYOffset - 70;
      window.scrollTo({ top: top, behavior: "smooth" });
    });
  });
})();
