
document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const inputs = {
        norte: document.getElementById('input-north'),
        sur: document.getElementById('input-south'),
        este: document.getElementById('input-east'),
        oeste: document.getElementById('input-west')
    };

    const valueDisplays = {
        norte: document.getElementById('val-north'),
        sur: document.getElementById('val-south'),
        este: document.getElementById('val-east'),
        oeste: document.getElementById('val-west')
    };

    const carsVisuals = {
        norte: document.getElementById('cars-north'),
        sur: document.getElementById('cars-south'),
        este: document.getElementById('cars-east'),
        oeste: document.getElementById('cars-west')
    };

    const lights = {
        0: document.getElementById('light-north'), // Norte
        1: document.getElementById('light-south'), // Sur
        2: document.getElementById('light-east'),  // Este
        3: document.getElementById('light-west')   // Oeste
    };

    const priorityResult = document.getElementById('priority-result');
    const liveToggle = document.getElementById('live-mode-toggle');

    let liveInterval = null;

    // Initialize
    updatePrediction();

    // Event Listeners
    Object.keys(inputs).forEach(key => {
        inputs[key].addEventListener('input', (e) => {
            valueDisplays[key].textContent = e.target.value;
            updateVisualDensity(key, e.target.value);
            if (!liveToggle.checked) {
                updatePrediction();
            }
        });
    });

    liveToggle.addEventListener('change', (e) => {
        if (e.target.checked) {
            startLiveMode();
        } else {
            stopLiveMode();
        }
    });

    function startLiveMode() {
        // Disable inputs
        Object.values(inputs).forEach(inp => inp.disabled = true);

        // Immediate update then interval
        fetchSimulation();
        liveInterval = setInterval(fetchSimulation, 2000);
    }

    function stopLiveMode() {
        clearInterval(liveInterval);
        Object.values(inputs).forEach(inp => inp.disabled = false);
    }

    function fetchSimulation() {
        fetch('/simulate')
            .then(res => res.json())
            .then(data => {
                // Update inputs
                inputs.norte.value = data.norte;
                inputs.sur.value = data.sur;
                inputs.este.value = data.este;
                inputs.oeste.value = data.oeste;

                // Update visuals
                Object.keys(data).forEach(key => {
                    valueDisplays[key].textContent = data[key];
                    updateVisualDensity(key, data[key]);
                });

                updatePrediction();
            });
    }

    function updateVisualDensity(direction, value) {
        // Opacity 0.1 to 1 based on value
        const opacity = 0.2 + (value / 100) * 0.8;
        carsVisuals[direction].style.opacity = opacity;
    }

    function updatePrediction() {
        const payload = {
            norte: inputs.norte.value,
            sur: inputs.sur.value,
            este: inputs.este.value,
            oeste: inputs.oeste.value
        };

        fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    console.error(data.error);
                    return;
                }
                setLights(data.prediction);
            })
            .catch(err => console.error(err));
    }

    function setLights(winnerIndex) {
        // Reset all lights
        document.querySelectorAll('.traffic-light .bulb.green').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.traffic-light .bulb.red').forEach(el => el.classList.add('active'));

        // Reset road highlights
        document.querySelectorAll('.road').forEach(el => el.classList.remove('active-road'));

        // Set Winner Light
        const winnerLight = lights[winnerIndex];
        if (winnerLight) {
            winnerLight.querySelector('.bulb.red').classList.remove('active');
            winnerLight.querySelector('.bulb.green').classList.add('active');

            const names = ['NORTE', 'SUR', 'ESTE', 'OESTE'];
            const roadClasses = ['.road.north', '.road.south', '.road.east', '.road.west'];

            // Highlight the Road Container
            const activeRoad = document.querySelector(roadClasses[winnerIndex]);
            if (activeRoad) activeRoad.classList.add('active-road');

            priorityResult.textContent = names[winnerIndex];
            priorityResult.style.color = 'var(--green-light)';
        }
    }
    // --- CV Mode Logic ---
    const cvStatus = document.getElementById('cv-toggle-status');
    const cvButton = document.getElementById('activate-cv'); // Keep button as valid alternative or remove
    let cvModeActive = false;

    // Toggle CV Mode via the "LIVE SIMULATION MODE" switch (User Request)
    liveToggle.addEventListener('change', () => {
        cvModeActive = liveToggle.checked;

        if (cvModeActive) {
            // Disable manual inputs
            Object.values(inputs).forEach(input => input.disabled = true);

            // Visual feedback
            if (cvButton) {
                cvButton.textContent = "DEACTIVATE TRAFFIC CONTROL";
                cvButton.classList.add('active');
            }
            if (cvStatus) {
                cvStatus.textContent = "ON - AUTO CONTROL";
                cvStatus.style.color = "#4ade80";
            }
        } else {
            // Enable manual inputs
            Object.values(inputs).forEach(input => input.disabled = false);

            if (cvButton) {
                cvButton.textContent = "ACTIVATE TRAFFIC CONTROL";
                cvButton.classList.remove('active');
            }
            if (cvStatus) {
                cvStatus.textContent = "OFF";
                cvStatus.style.color = "white";
            }
        }
    });

    // Also allow the big button to toggle the switch
    if (cvButton) {
        cvButton.addEventListener('click', () => {
            liveToggle.checked = !liveToggle.checked;
            liveToggle.dispatchEvent(new Event('change'));
        });
    }

    // Date Update
    const camDate = document.getElementById('cam-date');
    if (camDate) {
        setInterval(() => {
            const now = new Date();
            camDate.textContent = now.toLocaleDateString() + ' ' + now.toLocaleTimeString();
        }, 1000);
    }

    const originalUpdatePrediction = updatePrediction;
    updatePrediction = function () {
        if (!cvModeActive) {
            // Normal slider mode
            originalUpdatePrediction();
            return;
        }

        // CV Mode: Ask backend to use camera data
        const payload = {
            live_mode: true
        };

        fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    console.error(data.error);
                    return;
                }
                // Update slider visuals to match what CV sees
                if (data.traffic_data) {
                    // Update visual displays (spans)
                    Object.keys(data.traffic_data).forEach(key => {
                        // Check if matches input keys (norte/sur/este/oeste)
                        if (valueDisplays[key]) {
                            valueDisplays[key].textContent = data.traffic_data[key];
                            updateVisualDensity(key, data.traffic_data[key]);
                        }

                        // Update range inputs (for visual consistency)
                        if (inputs[key]) {
                            inputs[key].value = data.traffic_data[key];
                        }
                    });
                }
                setLights(data.prediction);
            })
            .catch(err => console.error(err));
    };
});
