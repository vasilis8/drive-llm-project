// ============================================
// Slide Navigation Logic
// ============================================

(function () {
    const slides = document.querySelectorAll('.slide');
    const counter = document.getElementById('slideCounter');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const total = slides.length;
    let current = 0;

    function goTo(index) {
        if (index < 0 || index >= total) return;

        const direction = index > current ? 'forward' : 'backward';

        slides[current].classList.remove('active');
        slides[current].classList.add(direction === 'forward' ? 'exit-left' : '');

        // Clean up transition classes after animation
        setTimeout(() => {
            slides[current === index ? 0 : current].classList.remove('exit-left');
        }, 500);

        current = index;

        // Reset incoming slide position
        slides[current].style.transform = direction === 'forward' ? 'translateX(60px)' : 'translateX(-60px)';
        slides[current].style.opacity = '0';

        // Force reflow
        void slides[current].offsetWidth;

        slides[current].style.transform = '';
        slides[current].style.opacity = '';
        slides[current].classList.add('active');

        counter.textContent = `${current + 1} / ${total}`;
    }

    function next() { goTo(current + 1); }
    function prev() { goTo(current - 1); }

    // Button clicks
    nextBtn.addEventListener('click', next);
    prevBtn.addEventListener('click', prev);

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowRight' || e.key === ' ') {
            e.preventDefault();
            next();
        }
        if (e.key === 'ArrowLeft') {
            e.preventDefault();
            prev();
        }
    });

    // Touch / swipe support
    let touchStartX = 0;
    document.addEventListener('touchstart', (e) => {
        touchStartX = e.changedTouches[0].screenX;
    });
    document.addEventListener('touchend', (e) => {
        const diff = touchStartX - e.changedTouches[0].screenX;
        if (Math.abs(diff) > 50) {
            diff > 0 ? next() : prev();
        }
    });
})();
