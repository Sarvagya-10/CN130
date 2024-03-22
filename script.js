const neighbourhoodSlider = document.getElementById('neighbourhood');
const connectivitySlider = document.getElementById('connectivity');
const electricitySlider = document.getElementById('electricity');

neighbourhoodSlider.addEventListener('input', function() {
    document.getElementById('neighbourhoodValue').textContent = this.value;
});

connectivitySlider.addEventListener('input', function() {
    document.getElementById('connectivityValue').textContent = this.value;
});

electricitySlider.addEventListener('input', function() {
    document.getElementById('electricityValue').textContent = this.value;
});