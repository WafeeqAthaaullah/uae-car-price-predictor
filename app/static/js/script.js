document.addEventListener("DOMContentLoaded", function() {
    
    // 1. Fetch dynamic options from the backend
    fetch('/options')
        .then(response => response.json())
        .then(data => {
            
            function formatChoices(items) {
                return items.map(item => ({ value: item, label: item }));
            }

            // A. Initialize the core dropdowns empty first
            const makeChoice = new Choices('#make', { searchEnabled: true, itemSelectText: '' });
            const modelChoice = new Choices('#model', { searchEnabled: true, itemSelectText: '' });
            const trimChoice = new Choices('#motors_trim', { searchEnabled: true, itemSelectText: '' });

            // B. Extract the nested car tree
            const carTree = data.car_tree;
            
            // C. Populate the Make dropdown with the first level of the tree
            const allMakes = Object.keys(carTree).sort();
            makeChoice.setChoices(formatChoices(allMakes), 'value', 'label', true);

            // D. The Cascade Logic: Listen for Make changes
            document.getElementById('make').addEventListener('change', function(e) {
                const selectedMake = e.target.value;
                
                // Erase whatever was in the Model and Trim boxes
                modelChoice.clearStore();
                trimChoice.clearStore();
                
                // Refill the Model box based on the Make selected
                if(carTree[selectedMake]) {
                    const validModels = Object.keys(carTree[selectedMake]).sort();
                    modelChoice.setChoices(formatChoices(validModels), 'value', 'label', true);
                }
            });

            // E. The Cascade Logic: Listen for Model changes
            document.getElementById('model').addEventListener('change', function(e) {
                const selectedMake = document.getElementById('make').value;
                const selectedModel = e.target.value;
                
                // Erase whatever was in the Trim box
                trimChoice.clearStore();
                
                // Refill the Trim box based on the exact Make + Model selected
                if(carTree[selectedMake] && carTree[selectedMake][selectedModel]) {
                    const validTrims = carTree[selectedMake][selectedModel].sort();
                    trimChoice.setChoices(formatChoices(validTrims), 'value', 'label', true);
                }
            });

            // Initialize Standard Dropdowns (No changes here)
            new Choices('#city', { choices: formatChoices(data.emirates), searchEnabled: false, itemSelectText: '' });
            new Choices('#body_type', { choices: formatChoices(data.body_types), searchEnabled: false, itemSelectText: '' });
            new Choices('#regional_specs', { choices: formatChoices(data.regional_specs), searchEnabled: false, itemSelectText: '' });
            new Choices('#fuel_type', { choices: formatChoices(data.fuel_types), searchEnabled: false, itemSelectText: '' });
            new Choices('#color', { choices: formatChoices(data.colors), searchEnabled: true, itemSelectText: '' });
            new Choices('#body_condition', { choices: formatChoices(data.body_conditions), searchEnabled: false, itemSelectText: '' });
            new Choices('#mechanical_condition', { choices: formatChoices(data.mechanical_conditions), searchEnabled: false, itemSelectText: '' });
            new Choices('#model_choice', { searchEnabled: false, itemSelectText: '' });
            
        })
        .catch(error => console.error("Error loading options:", error));

    // 2. Handle Form Submission
    document.getElementById('predictionForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const btn = document.getElementById('submitBtn');
        btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Analyzing Market Data...';
        btn.disabled = true;
        document.getElementById('resultBox').style.display = 'none';
        document.getElementById('errorBox').style.display = 'none';

        // Gather Payload
        const payload = {
            model_choice: document.getElementById('model_choice').value,
            make: document.getElementById('make').value,
            model: document.getElementById('model').value,
            motors_trim: document.getElementById('motors_trim').value,
            year: parseInt(document.getElementById('year').value) || 2020,
            mileage: parseInt(document.getElementById('mileage').value) || 50000,
            city: document.getElementById('city').value,
            body_type: document.getElementById('body_type').value,
            regional_specs: document.getElementById('regional_specs').value,
            cylinders: parseFloat(document.getElementById('cylinders').value),
            horsepower: parseFloat(document.getElementById('horsepower').value),
            transmission: document.getElementById('transmission').value,
            fuel_type: document.getElementById('fuel_type').value,
            color: document.getElementById('color').value,
            seller_type: document.getElementById('seller_type').value,
            body_condition: document.getElementById('body_condition').value,
            mechanical_condition: document.getElementById('mechanical_condition').value
        };

        // Send to Flask API
        fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(response => response.json())
        .then(data => {
            btn.innerHTML = 'Calculate Estimated Value';
            btn.disabled = false;

            if(data.predicted_price_aed) {
                const formattedPrice = data.predicted_price_aed.toLocaleString('en-US');
                document.getElementById('priceDisplay').innerText = `AED ${formattedPrice}`;
                document.getElementById('resultBox').style.display = 'block';
                document.getElementById('resultBox').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } else if (data.error) {
                document.getElementById('errorBox').innerText = "Error: " + data.error;
                document.getElementById('errorBox').style.display = 'block';
            }
        })
        .catch(error => {
            btn.innerHTML = 'Calculate Estimated Value';
            btn.disabled = false;
            document.getElementById('errorBox').innerText = "Connection failed. Is the Flask server running?";
            document.getElementById('errorBox').style.display = 'block';
        });
    });
});