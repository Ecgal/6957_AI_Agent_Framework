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