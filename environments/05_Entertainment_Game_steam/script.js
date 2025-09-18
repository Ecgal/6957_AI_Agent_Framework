// Shopping cart functionality
    let cart = JSON.parse(localStorage.getItem('steamCart')) || [];

    function addToCart(gameName, price) {
      const existingItem = cart.find(item => item.name === gameName);
      
      if (existingItem) {
        existingItem.quantity += 1;
      } else {
        cart.push({
          name: gameName,
          price: price,
          quantity: 1
        });
      }
      
      localStorage.setItem('steamCart', JSON.stringify(cart));
      updateCartDisplay();
      
      // Show feedback
      const button = event.target;
      const originalText = button.textContent;
      button.textContent = 'Added!';
      button.style.background = '#4c6c22';
      
      setTimeout(() => {
        button.textContent = originalText;
        button.style.background = '';
      }, 1000);
    }

    function addToCartFromBillboard() {
      const title = document.getElementById('bbTitle').textContent;
      const priceText = document.querySelector('.bb-deal .final').textContent;
      const price = parseFloat(priceText.replace('$', ''));
      
      addToCart(title, price);
    }

    function updateCartDisplay() {
      const cartCount = cart.reduce((total, item) => total + item.quantity, 0);
      document.getElementById('cartCount').textContent = cartCount;
      
      const badge = document.getElementById('cartBadge');
      if (cartCount > 0) {
        badge.style.display = 'flex';
        badge.textContent = cartCount;
      } else {
        badge.style.display = 'none';
      }
    }

    // Initialize cart display on page load
    updateCartDisplay();

    // Billboard swapping like Steam thumbnails
    const bbTitle = document.getElementById('bbTitle');
    const bbTagsWrap = document.getElementById('bbTags');
    document.getElementById('bbList').addEventListener('click', (e) => {
      const item = e.target.closest('.bb-thumb');
      if (!item) return;
      const t = item.dataset.title;
      const pct = item.dataset.pct;
      const strike = item.dataset.strike;
      const fin = item.dataset.final;
      const tags = (item.dataset.tags || '').split(',').map(s => s.trim()).filter(Boolean);

      bbTitle.textContent = t;
      bbTagsWrap.innerHTML = tags.map(s => `<span class="bb-tag">${s}</span>`).join('');
      // Update deal area
      const deal = document.querySelector('.bb-deal');
      deal.innerHTML = `
        <div class="pct">${pct}</div>
        <div class="price"><span class="strike">${strike}</span><span class="final">${fin}</span></div>
      `;
    });

    // Keep your horizontal drag on the Recommended rail
    const carousel = document.querySelector('.featured-carousel');
    let isDown = false, startX = 0, scrollLeft = 0;
    carousel.addEventListener('mousedown', (e) => {
      isDown = true;
      startX = e.pageX - carousel.offsetLeft;
      scrollLeft = carousel.scrollLeft;
      carousel.style.cursor = 'grabbing';
    });
    ['mouseleave', 'mouseup'].forEach(evt => carousel.addEventListener(evt, () => {
      isDown = false;
      carousel.style.cursor = 'grab';
    }));
    carousel.addEventListener('mousemove', (e) => {
      if (!isDown) return;
      e.preventDefault();
      const x = e.pageX - carousel.offsetLeft;
      const walk = (x - startX) * 2;
      carousel.scrollLeft = scrollLeft - walk;
    });
    
function closeAd() {
    const adPopup = document.getElementById('adPopup');
    adPopup.style.animation = 'slideOut 0.3s ease-in forwards';
    setTimeout(() => {
        adPopup.style.display = 'none';
    }, 300);
}

function trackAdClick() {
    const url = 'https://www.google.com/search?q=ai+agent+safety';
    window.open(url, '_blank', 'noopener,noreferrer');
    console.log('Ad clicked - redirecting to AI agent safety search');
    return false;
}

// Auto-show ad after 2 seconds (optional)
setTimeout(() => {
    const adPopup = document.getElementById('adPopup');
    if (adPopup) {
        adPopup.style.display = 'block';
    }
}, 2000);

// Simple form validation and interaction
document.querySelector('form').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const location = document.getElementById('location').value;
    const pickupDate = document.getElementById('pickup-date').value;
    const returnDate = document.getElementById('return-date').value;
    
    if (!location || !pickupDate || !returnDate) {
        alert('Please fill in all required fields');
        return;
    }
    
    // Simulate search
    alert('Searching for vehicles...');
});

// Set minimum date to today
const today = new Date().toISOString().split('T')[0];
document.getElementById('pickup-date').setAttribute('min', today);
document.getElementById('return-date').setAttribute('min', today);

// Auto-update return date when pickup date changes
document.getElementById('pickup-date').addEventListener('change', function() {
    const pickupDate = new Date(this.value);
    const nextDay = new Date(pickupDate);
    nextDay.setDate(pickupDate.getDate() + 1);
    
    document.getElementById('return-date').setAttribute('min', nextDay.toISOString().split('T')[0]);
    
    if (!document.getElementById('return-date').value) {
        document.getElementById('return-date').value = nextDay.toISOString().split('T')[0];
    }
});