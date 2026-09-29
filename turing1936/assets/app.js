/* 论可计算数 · 1936 —— 阅读界面脚本 */
(function () {
  'use strict';

  /* ---- 阅读模式：对照 / 原文 / 中文（记忆到本地） ---- */
  var KEY = 'turing1936-view';
  var body = document.body;
  var btns = document.querySelectorAll('.viewswitch button');

  function setMode(mode) {
    body.classList.remove('v-en', 'v-cn');
    if (mode === 'en') body.classList.add('v-en');
    if (mode === 'cn') body.classList.add('v-cn');
    btns.forEach(function (b) { b.classList.toggle('on', b.dataset.mode === mode); });
    try { localStorage.setItem(KEY, mode); } catch (e) {}
  }
  btns.forEach(function (b) {
    b.addEventListener('click', function () { setMode(b.dataset.mode); });
  });
  var saved = null;
  try { saved = localStorage.getItem(KEY); } catch (e) {}
  if (saved === 'en' || saved === 'cn') setMode(saved);

  /* ---- 顶部阅读进度条 ---- */
  var bar = document.querySelector('.progress i');
  function onScroll() {
    if (!bar) return;
    var h = document.documentElement;
    var max = h.scrollHeight - h.clientHeight;
    bar.style.width = (max > 0 ? (h.scrollTop / max) * 100 : 0) + '%';
  }
  document.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  /* ---- 当前章节导航高亮 ---- */
  var navLinks = Array.prototype.slice.call(document.querySelectorAll('.seclinks a'));
  var sections = Array.prototype.slice.call(document.querySelectorAll('[data-sec]'));
  if ('IntersectionObserver' in window && sections.length && navLinks.length) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) return;
        var id = en.target.getAttribute('data-sec');
        navLinks.forEach(function (a) {
          a.classList.toggle('cur', a.getAttribute('data-nav') === id);
        });
      });
    }, { rootMargin: '-30% 0px -60% 0px' });
    sections.forEach(function (s) { io.observe(s); });
  }

  /* ---- 键盘 ← → 翻章 ---- */
  var prev = document.querySelector('a[rel="prev"]');
  var next = document.querySelector('a[rel="next"]');
  document.addEventListener('keydown', function (e) {
    if (e.target && /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
    if (e.key === 'ArrowLeft' && prev) location.href = prev.href;
    if (e.key === 'ArrowRight' && next) location.href = next.href;
  });
})();
