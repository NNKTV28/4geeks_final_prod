/* ══════════════════════════════════════════════════════════════════════════
   MovieLens 100K — Presentación Interactiva
   Script
   ══════════════════════════════════════════════════════════════════════════ */

(function () {
  "use strict";

  // ── DOM refs ─────────────────────────────────────────────────────────
  const slides      = document.querySelectorAll(".slide");
  const progressBar = document.getElementById("progress-bar");
  const counter     = document.getElementById("slide-counter");
  const btnPrev     = document.getElementById("btn-prev");
  const btnNext     = document.getElementById("btn-next");
  const sidebar     = document.getElementById("sidebar");
  const sidebarBtn  = document.getElementById("sidebar-toggle");
  const navList     = document.getElementById("nav-list");
  const badge       = document.getElementById("speaker-badge");
  const modal       = document.getElementById("img-modal");
  const modalImg    = document.getElementById("modal-img");
  const modalClose  = document.getElementById("modal-close");

  let current = 0;
  const total = slides.length;

  // ── Build navigation ─────────────────────────────────────────────────
  slides.forEach((s, i) => {
    const li = document.createElement("li");
    const title = s.dataset.title || `Slide ${i + 1}`;
    const speaker = s.dataset.speaker || "";
    let html = `<span>${i + 1}. ${title}</span>`;
    if (speaker) {
      const cls = speaker.toLowerCase() === "nikita" ? "speaker-nikita" : "speaker-paloma";
      html += `<span class="nav-speaker ${cls}">${speaker}</span>`;
    }
    li.innerHTML = html;
    li.addEventListener("click", () => goTo(i));
    navList.appendChild(li);
  });

  // ── Navigation ───────────────────────────────────────────────────────
  function goTo(index) {
    if (index < 0 || index >= total || index === current) return;
    const direction = index > current ? 1 : -1;

    slides[current].classList.remove("active");
    slides[current].classList.toggle("prev", direction === 1);

    current = index;

    slides[current].classList.remove("prev");
    slides[current].classList.add("active");

    updateUI();
    animateNumbers();
  }

  function next() { goTo(current + 1); }
  function prev() { goTo(current - 1); }

  function updateUI() {
    // progress
    progressBar.style.width = ((current / (total - 1)) * 100) + "%";
    // counter
    counter.textContent = `${current + 1} / ${total}`;
    // nav highlight
    navList.querySelectorAll("li").forEach((li, i) => {
      li.classList.toggle("active", i === current);
    });
    // speaker badge
    const speaker = slides[current].dataset.speaker || "";
    if (speaker) {
      badge.textContent = `🎤 ${speaker}`;
      badge.className = `visible ${speaker.toLowerCase()}`;
    } else {
      badge.className = "";
    }
  }

  // ── Animated counters ────────────────────────────────────────────────
  function animateNumbers() {
    const nums = slides[current].querySelectorAll(".stat-number[data-target]");
    nums.forEach((el) => {
      const target = parseInt(el.dataset.target, 10);
      if (isNaN(target)) return;
      let start = 0;
      const duration = 1200;
      const step = Math.ceil(target / (duration / 16));
      function tick() {
        start += step;
        if (start >= target) {
          el.textContent = target.toLocaleString("es-ES");
          return;
        }
        el.textContent = start.toLocaleString("es-ES");
        requestAnimationFrame(tick);
      }
      el.textContent = "0";
      requestAnimationFrame(tick);
    });
  }

  // ── Keyboard ─────────────────────────────────────────────────────────
  document.addEventListener("keydown", (e) => {
    if (modal.classList.contains("open")) {
      if (e.key === "Escape") modal.classList.remove("open");
      return;
    }
    switch (e.key) {
      case "ArrowRight":
      case "PageDown":
      case " ":
        e.preventDefault();
        next();
        break;
      case "ArrowLeft":
      case "PageUp":
        e.preventDefault();
        prev();
        break;
      case "Home":
        e.preventDefault();
        goTo(0);
        break;
      case "End":
        e.preventDefault();
        goTo(total - 1);
        break;
      case "Escape":
        sidebar.classList.remove("open");
        break;
    }
  });

  // ── Buttons ──────────────────────────────────────────────────────────
  btnNext.addEventListener("click", next);
  btnPrev.addEventListener("click", prev);
  sidebarBtn.addEventListener("click", () => sidebar.classList.toggle("open"));

  // ── Touch/swipe ──────────────────────────────────────────────────────
  let touchStartX = 0;
  document.addEventListener("touchstart", (e) => {
    touchStartX = e.changedTouches[0].clientX;
  }, { passive: true });
  document.addEventListener("touchend", (e) => {
    const dx = e.changedTouches[0].clientX - touchStartX;
    if (Math.abs(dx) > 50) {
      dx < 0 ? next() : prev();
    }
  }, { passive: true });

  // ── Image zoom modal ─────────────────────────────────────────────────
  document.addEventListener("click", (e) => {
    if (e.target.classList.contains("zoomable")) {
      modalImg.src = e.target.src;
      modal.classList.add("open");
    }
  });
  modal.addEventListener("click", () => modal.classList.remove("open"));
  modalClose.addEventListener("click", () => modal.classList.remove("open"));

  // ── Init ─────────────────────────────────────────────────────────────
  updateUI();
})();
