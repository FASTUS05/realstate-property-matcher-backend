// static/script.js
document.addEventListener('DOMContentLoaded', function() {
    const saleUrlInput = document.getElementById('saleUrlInput');
    const matchButton = document.getElementById('matchButton');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultsContainer = document.getElementById('resultsContainer');
    const salePropertyInfo = document.getElementById('salePropertyInfo');
    const matchesList = document.getElementById('matchesList');
    const errorMessage = document.getElementById('errorMessage');
    const exportCsvButton = document.getElementById('exportCsvButton');
    const exportPdfButton = document.getElementById('exportPdfButton');

    let currentMatchesData = null; // To store data for export

    matchButton.addEventListener('click', async function() {
        const saleUrl = saleUrlInput.value.trim();
        if (!saleUrl) {
            displayError('Please enter a valid property for sale URL.');
            return;
        }

        resetUI();
        loadingSpinner.style.display = 'block';
        matchButton.disabled = true;

        try {
            const response = await fetch('/match', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ sale_url: saleUrl }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            currentMatchesData = await response.json();
            displayResults(currentMatchesData);

        } catch (error) {
            console.error('Error:', error);
            displayError(`Failed to fetch matches: ${error.message}`);
        } finally {
            loadingSpinner.style.display = 'none';
            matchButton.disabled = false;
        }
    });

    function displayResults(data) {
        if (data.matches && data.matches.length > 0) {
            resultsContainer.style.display = 'block';
            
            // Display Sale Property Info
            salePropertyInfo.innerHTML = `
                <h2>Property for Sale Details</h2>
                <p><strong>Title:</strong> ${data.sale_listing.title}</p>
                <p><strong>Description:</strong> ${data.sale_listing.desc.substring(0, 200)}${data.sale_listing.desc.length > 200 ? '...' : ''}</p>
                <p><strong>Price:</strong> ${formatPrice(data.sale_listing.price)}</p>
                <p><strong>Rooms:</strong> ${data.sale_listing.rooms}</p>
                <p><strong>Location:</strong> ${data.sale_listing.location}</p>
                <div class="images">
                    ${data.sale_listing.images.map(img => `<img src="${img}" alt="Sale Property Image">`).join('')}
                </div>
            `;


            matchesList.innerHTML = ''; // Clear previous results
            data.matches.forEach(match => {
                const matchCard = document.createElement('div');
                matchCard.classList.add('match-card');
                matchCard.innerHTML = `
                    <img src="${match.image}" alt="${match.title}">
                    <h3>${match.title}</h3>
                    <p><strong>Platform:</strong> ${match.platform}</p>
                    <p><strong>Location:</strong> ${match.location}</p>
                    <p><strong>Price:</strong> ${formatPrice(match.price)}</p>
                    <p><strong>Rooms:</strong> ${match.rooms}</p>
                    <p class="similarity-score">Similarity: ${match.final_score}%</p>
                    <a href="${match.url}" target="_blank" class="platform-link">View Listing</a>
                `;
                matchesList.appendChild(matchCard);
            });
        } else {
            resultsContainer.style.display = 'none';
            displayError('No matching rentals found. Try a different sale URL.');
        }
    }

    function formatPrice(price) {
        if (typeof price === 'number') {
            return `PKR ${price.toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0})}`;
        }
        return price; // Return as is if not a number
    }

    function displayError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
    }

    function resetUI() {
        errorMessage.style.display = 'none';
        resultsContainer.style.display = 'none';
        salePropertyInfo.innerHTML = '';
        matchesList.innerHTML = '';
        currentMatchesData = null;
    }

    exportCsvButton.addEventListener('click', async function() {
        if (!currentMatchesData) {
            displayError("No data to export. Please find matches first.");
            return;
        }
        try {
            const response = await fetch('/export_csv', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(currentMatchesData),
            });
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = 'rental_matches.csv';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
            } else {
                const errorText = await response.text();
                throw new Error(`CSV export failed: ${errorText}`);
            }
        } catch (error) {
            console.error('Export CSV error:', error);
            displayError(`Failed to export CSV: ${error.message}`);
        }
    });

    exportPdfButton.addEventListener('click', async function() {
        if (!currentMatchesData) {
            displayError("No data to export. Please find matches first.");
            return;
        }
        try {
            const response = await fetch('/export_pdf', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(currentMatchesData),
            });
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = 'rental_matches.pdf';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
            } else {
                const errorText = await response.text();
                throw new Error(`PDF export failed: ${errorText}`);
            }
        } catch (error) {
            console.error('Export PDF error:', error);
            displayError(`Failed to export PDF: ${error.message}`);
        }
    });
});