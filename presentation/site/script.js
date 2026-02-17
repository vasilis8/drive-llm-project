/* ==============================================
   DriveLLM — Project Website Script
   Scroll-based reveal animations + smooth nav
   ============================================== */

document.addEventListener('DOMContentLoaded', () => {

    // ── Scroll-based fade-in ────────────────────────────────
    const observerOptions = {
        root: null,
        rootMargin: '0px 0px -60px 0px',
        threshold: 0.1,
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Tag elements for animation
    const selectors = [
        '.section h2',
        '.section-label',
        '.section-desc',
        '.stat-card',
        '.arch-card',
        '.fusion-block',
        '.pipe-step',
        '.result-card',
        '.cmd-category',
        '.download-card',
        '.about-text',
        '.tech-stack',
    ];

    selectors.forEach(sel => {
        document.querySelectorAll(sel).forEach((el, i) => {
            el.classList.add('fade-in');
            el.style.transitionDelay = `${i * 0.06}s`;
            observer.observe(el);
        });
    });

    // ── Navbar background on scroll ─────────────────────────
    const nav = document.getElementById('nav');
    const scrollIndicator = document.getElementById('scrollIndicator');

    window.addEventListener('scroll', () => {
        const scrolled = window.scrollY > 80;
        if (scrolled) {
            nav.style.borderBottomColor = 'rgba(255,255,255,0.08)';
        } else {
            nav.style.borderBottomColor = 'rgba(255,255,255,0.04)';
        }

        // Hide scroll indicator
        if (scrollIndicator) {
            scrollIndicator.style.opacity = window.scrollY > 100 ? '0' : '0.3';
        }
    }, { passive: true });

    // ── Smooth scroll for anchor links ──────────────────────
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', (e) => {
            e.preventDefault();
            const target = document.querySelector(anchor.getAttribute('href'));
            if (target) {
                const navHeight = nav.offsetHeight;
                const targetPos = target.getBoundingClientRect().top + window.scrollY - navHeight;
                window.scrollTo({ top: targetPos, behavior: 'smooth' });
            }
        });
    });

});
