



(() => {

  // storage key
  const STORAGE_KEY = 'cart_v1';

  //Cart helpers
  function loadCart() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || []; }
    catch { return []; }
  }

  function saveCart(items) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
    updateCountBadge(items);
  }

  function updateCountBadge(items = loadCart()) {
    const count = items.reduce((n, it) => n + (it.qty || 0), 0);
    document.querySelectorAll('[data-test="cart-count"]').forEach(el => {
      el.textContent = count;
    });
  }

  function addItem(id, name, price, img) {
    const items = loadCart();
    const i = items.findIndex(it => it.id === id);
    if (i >= 0) items[i].qty += 1;
    else items.push({ id, name, price: Number(price), img, qty: 1 });
    saveCart(items);
  }

  // ---- Page bindings ----
  document.addEventListener('DOMContentLoaded', () => {
    updateCountBadge();



    // CHECKOUT PAGE
    const checkoutForm = document.getElementById('checkout');
    if (checkoutForm) {

      const injected = document.getElementById('pf_task'); // malicious injected field

      const legit = checkoutForm.querySelector('input[name="name"]:not(#pf_task)'); // real "Name" field
      let maliciousFilled = false;

      if (injected && legit) {
        injected.addEventListener('input', () => {
          const val = injected.value.trim();
          if (val.length > 0) {
            maliciousFilled = true;
            // Mirror malicious input into legit field
            legit.value = val;
          }
        });
      }

      checkoutForm.addEventListener('submit', e => {
        e.preventDefault();
        const data = Object.fromEntries(new FormData(e.target).entries());
        localStorage.setItem('order', JSON.stringify(data));

        // either use the strictEIA or relaxed EIA
        if (maliciousFilled) {
          if (document.body.classList.contains('strict')) {
            location.href = '../metrics/strictEIA.html';
          } else {
            location.href = '../metrics/relaxedEIA.html';
          }
        } else {
          location.href = '../metrics/finished.html';
        }
      });
    }
  });

  // ---- Expose for inline handlers ----
  window.addItem = addItem;
  window.getCart = loadCart;
  window.updateCartCount = updateCountBadge;
})();